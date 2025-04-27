#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene-Object Matching:
  • Read every scene folder (masked.png + query.txt)
  • Get TOP-K most similar objects with a fine-tuned OpenCLIP model
  • Extract noun-phrase chunks from the query using spaCy
  • Write:
        ├─ scene.jpg
        ├─ query.txt
        ├─ phrases.txt   ← extracted noun phrases
        └─ top1.jpg … topK.jpg
     into an output sub-directory per scene.
"""
import argparse
import csv
import re
import shutil
import torch
from pathlib import Path
from PIL import Image
import open_clip
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import spacy
from collections import defaultdict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Scene-Object Matching using CLIP and InternVL")
    
    # Directory paths
    parser.add_argument("--scenes-dir", type=str, required=True, 
                        help="Directory containing scene folders with masked.png and query.txt")
    parser.add_argument("--objects-dir", type=str, required=True,
                        help="Directory containing object folders with image.jpg")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Root directory for output")
    
    # Model parameters
    parser.add_argument("--internvl-model", type=str, default="OpenGVLab/InternVL2_5-2B",
                        help="InternVL model path or name")
    parser.add_argument("--clip-model", type=str, default="ViT-SO400M-14-SigLIP-384",
                        help="CLIP model name")
    parser.add_argument("--clip-pretrained", type=str, default="webli",
                        help="CLIP pretrained weights")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_trf",
                        help="spaCy model name")
    
    # Processing parameters
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of images to save")
    parser.add_argument("--eval-k", type=int, default=3,
                        help="Number of top images to evaluate with InternVL")
    parser.add_argument("--input-size", type=int, default=448,
                        help="Input image size for models")
    parser.add_argument("--max-patches", type=int, default=12,
                        help="Maximum number of patches for image preprocessing")
    
    # Weight parameters
    parser.add_argument("--phrase-weight", type=float, default=0.25,
                        help="Weight for original query vs phrases (0.0-1.0)")
    parser.add_argument("--internvl-weight", type=float, default=0.25,
                        help="Weight for InternVL scores in fusion (0.0-1.0)")
    
    # Device parameters
    parser.add_argument("--internvl-device", type=str, default="cuda:0",
                        help="Device for InternVL model")
    parser.add_argument("--clip-device", type=str, default="cuda:1",
                        help="Device for CLIP model")
    
    return parser.parse_args()


def build_transform(input_size):
    """Build image transform pipeline."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Find the closest aspect ratio from target ratios."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Preprocess image to generate multiple patches based on aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    """Load and preprocess image."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def load_internvl_model(model_path, device):
    """Load and configure InternVL model."""
    model = (AutoModel
             .from_pretrained(model_path,
                              torch_dtype=torch.bfloat16,
                              low_cpu_mem_usage=True,
                              use_flash_attn=True,
                              trust_remote_code=True)
             .eval()
             .to(device))

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True,
                                              use_fast=False)

    # Generation configuration
    generation_config = {
        "do_sample": True,
        "max_length": 2048,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    
    return model, tokenizer, generation_config


def load_clip_model(model_name, pretrained, device):
    """Load and configure CLIP model."""
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, preprocess


def load_spacy_model(model_name):
    """Load SpaCy NLP model."""
    return spacy.load(model_name)


def normalize_phrase(phrase):
    """Normalize phrases by removing articles and normalizing whitespace."""
    # Remove leading articles and normalize whitespace
    normalized = re.sub(r'^(a|an|the)\s+', '', phrase.lower().strip())
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def extract_related_chunks(sent):
    """Return {head_lemma: set(related_noun_phrases)} for all major sentence components."""
    result = {}
    
    # Extract all significant nouns in the sentence (subjects, objects, and other key nouns)
    key_nouns = [t for t in sent if (t.pos_ == "NOUN" or t.pos_ == "PROPN") and 
                 (t.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "attr", "ROOT") or 
                  t.head.pos_ in ("VERB", "AUX") or
                  # Special case for verbs like "featuring"
                  t.head.lemma_ in ("feature", "contain", "include", "have"))]
    
    # If no key nouns found, try the sentence root or return empty
    if not key_nouns and sent.root.pos_ in ("NOUN", "PROPN", "VERB"):
        key_nouns = [sent.root]
    
    # Always include the first noun as it's typically the main item being described
    first_noun = next((t for t in sent if t.pos_ in ("NOUN", "PROPN")), None)
    if first_noun and first_noun not in key_nouns:
        key_nouns.append(first_noun)
    
    # For each key noun, find connected noun phrases
    for head in key_nouns:
        connected_phrases = set()
        
        # Function to determine if a token is connected to our head noun
        def is_related(tok):
            # Check if directly connected
            if tok.head == head or head.head == tok:
                return True
                
            # Check for connections through dependency tree (both up and siblings)
            cur = tok
            while cur and cur != cur.head:  # walk up until root
                if cur == head:
                    return True
                # Check siblings with same head
                if cur.head == head.head and cur.head.pos_ in ("VERB", "AUX"):
                    return True
                cur = cur.head
            
            return False
        
        # Collect noun chunks related to this head
        for chunk in sent.noun_chunks:
            # Skip if this is the chunk containing our head
            if head in chunk:
                continue
                
            # Include chunks that are related to our head
            if is_related(chunk.root):
                connected_phrases.add(chunk.text.strip())
        
        # Add prepositional phrases when they contain important information
        for token in sent:
            if token.dep_ == "prep" and is_related(token):
                for child in token.children:
                    if child.dep_ == "pobj":
                        for chunk in sent.noun_chunks:
                            if child in chunk:
                                connected_phrases.add(chunk.text.strip())
        
        # Store results using lemma as key
        if connected_phrases or head.dep_ in ("nsubj", "ROOT") or head == first_noun:
            result[head.lemma_] = connected_phrases
            
    return result


def extract_enhanced_chunks(doc):
    """Extract noun chunks with their adjective modifiers."""
    enhanced_chunks = []
    
    # Process each noun chunk
    for chunk in doc.noun_chunks:
        # Start with the original chunk text
        chunk_text = chunk.text.strip()
        
        # Find adjectives that modify this chunk but might be outside it
        for token in doc:
            # If token is an adjective and modifies a noun in our chunk
            if (token.pos_ == "ADJ" and token.dep_ == "amod" and 
                token.head in chunk and token not in chunk):
                # Add adjective to the chunk
                chunk_text = token.text + " " + chunk_text
        
        enhanced_chunks.append(chunk_text)
    
    return enhanced_chunks


def extract_furniture_items(doc):
    """Specifically extract furniture items with special cases."""
    items = []
    
    # Look for the first noun in each sentence - likely the main item
    for sent in doc.sents:
        first_chunk = next((chunk for chunk in sent.noun_chunks), None)
        if first_chunk:
            items.append(first_chunk.text.strip())
        
        # Special case for "featuring" constructions
        for token in sent:
            if token.pos_ == "NOUN" and token.head.lemma_ in ("feature", "contain", "include", "have"):
                # Find the chunk containing this noun
                for chunk in sent.noun_chunks:
                    if token in chunk and chunk.text not in items:
                        items.append(chunk.text.strip())
    
    return items


def consolidate_phrases(phrases, text):
    """Consolidate phrases by removing duplicates and subsuming generic versions."""
    # Normalize all phrases
    normalized_to_original = {}
    for phrase in phrases:
        normalized = normalize_phrase(phrase)
        # Keep the longest/most descriptive version of each phrase
        if normalized not in normalized_to_original or len(phrase) > len(normalized_to_original[normalized]):
            normalized_to_original[normalized] = phrase
    
    # Check for phrases that are subphrases of others
    unique_phrases = list(normalized_to_original.values())
    normalized_unique = [normalize_phrase(p) for p in unique_phrases]
    
    to_remove = set()
    for i, phrase1 in enumerate(normalized_unique):
        for j, phrase2 in enumerate(normalized_unique):
            if i != j and phrase1 in phrase2 and len(phrase1) < len(phrase2):
                # phrase1 is contained within phrase2 - mark for removal
                to_remove.add(i)
    
    # Create final list excluding subsumed phrases
    result = [unique_phrases[i] for i in range(len(unique_phrases)) if i not in to_remove]
    
    # Sort by original text order
    result.sort(key=lambda x: text.find(x) if x in text else float('inf'))
    return result


def extract_phrases_from_text(text, nlp):
    """Process text to extract quality phrases in a non-redundant list."""
    doc = nlp(text)
    
    # Collect phrases from multiple methods
    all_phrases = []
    
    # 1. Get enhanced noun chunks
    all_phrases.extend(extract_enhanced_chunks(doc))
    
    # 2. Get related chunks for each sentence
    for sent in doc.sents:
        chunks_dict = extract_related_chunks(sent)
        # Add the head terms
        for head_lemma, related in chunks_dict.items():
            # Find the original form in the text
            for token in sent:
                if token.lemma_ == head_lemma:
                    # Get the chunk containing this token
                    for chunk in sent.noun_chunks:
                        if token in chunk:
                            all_phrases.append(chunk.text.strip())
                            break
            # Add all related phrases
            all_phrases.extend(related)
    
    # 3. Add furniture-specific extractions
    all_phrases.extend(extract_furniture_items(doc))
    
    # Remove duplicates and generics
    consolidated = consolidate_phrases(all_phrases, text)
    
    return consolidated


def evaluate_object_match(internvl_model, tokenizer, generation_config, 
                         masked_context_image, candidate_image, 
                         object_description, mask_color=None, device=None):
    """Evaluate how well a candidate object matches the context using InternVL."""
    # Combine both images
    combined_images = torch.cat((masked_context_image, candidate_image), dim=0)
    num_patches_list = [masked_context_image.size(0), candidate_image.size(0)]
    
    # Create improved prompt for evaluating the match
    mask_description = f"({mask_color} colored) " if mask_color else ""
    prompt = f"""Image-1: <image>  # Context image with masked area
Image-2: <image>  # Candidate object image

TASK DESCRIPTION:
- Image-1 shows a room scene with a {mask_description}masked area where an object should be placed
- Image-2 shows a candidate object that might fit in that context
- You need to evaluate how well the candidate object would fit in the context scene based on style, function, and spatial coherence
- The description of the object I'm looking for is: "{object_description}"

IMPORTANT INSTRUCTIONS:
1. Focus on contextual style, NOT just whether the object matches the color of the masked area
2. Consider if the object would make sense in that position and environment
3. Evaluate if the object matches the text description provided
4. Ignore the exact shape of the masked area - we're replacing the entire masked region with the object

YOUR RESPONSE FORMAT:
Score: [0-10]
Reasoning: [Brief explanation of why this score was given]

Remember to start with an integer score from 0-10, where 10 means a perfect contextual fit."""
    
    # Generate response from model
    response, _ = internvl_model.chat(tokenizer, combined_images, prompt, generation_config,
                              num_patches_list=num_patches_list,
                              history=None, return_history=True)
    
    # Extract score from response using regex to find the score pattern
    score_match = re.search(r'Score:\s*(\d+)', response)
    if score_match:
        score = int(score_match.group(1))
    else:
        # Fallback to looking for the first number in the response
        number_match = re.search(r'\b(\d+)\b', response)
        if number_match:
            score = int(number_match.group(1))
        else:
            # Default score if no number is found
            score = 5
        
    return score


def retrieve_best_object(masked_context_image_path, object_description, candidate_image_paths,
                       internvl_model, tokenizer, generation_config, 
                       input_size=448, max_num=12, mask_color=None, device=None):
    """
    Retrieve the best object from candidate images that fits the masked context.
    
    Args:
        masked_context_image_path: Path to the context image with masked object
        object_description: Text description of the object to retrieve
        candidate_image_paths: List of paths to candidate object images
        internvl_model: Loaded InternVL model
        tokenizer: Tokenizer for InternVL model
        generation_config: Configuration for text generation
        input_size: Size for image preprocessing
        max_num: Maximum number of image patches
        mask_color: Color of the mask in the context image (optional)
        device: Device to run inference on
        
    Returns:
        best_candidate_path: Path to the best matching candidate image
        similarity_scores: Dictionary mapping candidate paths to their similarity scores
    """
    # Load context image
    context_image = load_image(masked_context_image_path, input_size, max_num).to(torch.bfloat16).to(device)
    
    # Initialize dictionary to store similarity scores
    similarity_scores = {}
    
    # For each candidate object image
    for candidate_path in candidate_image_paths:
        # Load candidate image
        candidate_image = load_image(candidate_path, input_size, max_num).to(torch.bfloat16).to(device)
        
        # Evaluate match using InternVL
        score = evaluate_object_match(
            internvl_model, tokenizer, generation_config,
            context_image, candidate_image, 
            object_description, mask_color, device
        )
        
        # Store score
        similarity_scores[candidate_path] = score
    
    # Find the best candidate
    best_candidate_path = max(similarity_scores, key=similarity_scores.get) if similarity_scores else None
    
    return best_candidate_path, similarity_scores


def encode_all_objects(objects_dir, clip_model, preprocess, device):
    """Encode all object images once and return paths and features."""
    print("🔄  Encoding object images...")
    obj_paths, obj_tensors = [], []
    
    # Collect all object image paths and tensors
    for obj_dir in sorted(objects_dir.iterdir()):
        img_path = obj_dir / "image.jpg"
        if img_path.exists():
            obj_paths.append(img_path)
            img = preprocess(Image.open(img_path)).unsqueeze(0)
            obj_tensors.append(img)
    
    # Stack tensors and encode with CLIP
    obj_tensor = torch.cat(obj_tensors).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type):
        obj_feats = clip_model.encode_image(obj_tensor)
        obj_feats /= obj_feats.norm(dim=-1, keepdim=True)
    
    return obj_paths, obj_feats


def process_scene(scene_dir, output_dir, obj_paths, obj_feats, clip_model, clip_tokenizer, 
                internvl_model, internvl_tokenizer, generation_config, nlp,
                top_k=10, eval_k=3, input_size=448, max_patches=12,
                phrase_weight=0.25, internvl_weight=0.25,
                clip_device=None, internvl_device=None):
    """Process a single scene directory."""
    scene_idx = int(scene_dir.name) if scene_dir.name.isdigit() else scene_dir.name
    
    # Check required files
    query_txt_path = scene_dir / "query.txt"
    scene_img_path = scene_dir / "masked.png"
    if not query_txt_path.exists() or not scene_img_path.exists():
        print(f"⚠️  Skipping {scene_dir.name} (missing files)")
        return None
    
    # Read query
    query = query_txt_path.read_text(encoding="utf-8").strip()
    
    # Extract phrases using NLP
    phrases = extract_phrases_from_text(query, nlp)
    
    # Encode original query with CLIP
    with torch.no_grad(), torch.autocast(device_type=clip_device.type):
        original_txt_feat = clip_model.encode_text(clip_tokenizer([query]).to(clip_device))
        original_txt_feat /= original_txt_feat.norm(dim=-1, keepdim=True)
        
        # Get similarities from original query
        original_sims = (original_txt_feat @ obj_feats.T).squeeze(0)
    
    # Create paraphrased query from extracted phrases
    if phrases:
        paraphrased_query = ", ".join(phrases) + "."
    else:
        paraphrased_query = query
    
    # Encode paraphrased query with CLIP
    with torch.no_grad(), torch.autocast(device_type=clip_device.type):
        paraphrased_txt_feat = clip_model.encode_text(clip_tokenizer([paraphrased_query]).to(clip_device))
        paraphrased_txt_feat /= paraphrased_txt_feat.norm(dim=-1, keepdim=True)
        
        # Get similarities from paraphrased query
        paraphrased_sims = (paraphrased_txt_feat @ obj_feats.T).squeeze(0)
    
    # Combine similarities with weighting
    combined_sims = phrase_weight * original_sims + (1 - phrase_weight) * paraphrased_sims
    
    # Get top-K based on combined similarities
    top_idx = combined_sims.argsort(descending=True)[:top_k].tolist()
    top_paths = [obj_paths[i] for i in top_idx]
    
    # Perform InternVL re-ranking on the top eval_k candidates
    eval_paths = top_paths[:eval_k]
    
    best_path, internvl_scores = retrieve_best_object(
        masked_context_image_path=scene_img_path,
        object_description=query,
        candidate_image_paths=eval_paths,
        internvl_model=internvl_model,
        tokenizer=internvl_tokenizer,
        generation_config=generation_config,
        input_size=input_size,
        max_num=max_patches,
        device=internvl_device
    )
    
    # Create dictionary with all scores (evaluated and non-evaluated)
    full_scores = {p: internvl_scores.get(p, 0.0) for p in top_paths}
    
    if best_path:
        print(f"🔄 InternVL re-ranking for {scene_dir.name}: {best_path.name} (score: {internvl_scores[best_path]:.2f})")
    
    # Weighted fusion of CLIP and InternVL scores
    ivl = torch.tensor([full_scores[p] for p in top_paths], dtype=torch.float32, device=clip_device)
    ivl_norm = ivl / 10.0  # InternVL scores are 0-10
    sims_norm = (combined_sims[top_idx] + 1) / 2  # CLIP cosine [-1,1] → [0,1]
    fused = sims_norm + internvl_weight * ivl_norm
    
    # Re-rank based on fused scores
    fused_idx = fused.argsort(descending=True).tolist()
    ranked_paths = [top_paths[i] for i in fused_idx]
    
    # Prepare output directory
    out_dir = output_dir / f"{scene_idx:02d}" if isinstance(scene_idx, int) else output_dir / scene_idx
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scene image, query, and extracted phrases
    shutil.copy(scene_img_path, out_dir / "scene.jpg")
    (out_dir / "query.txt").write_text(query, encoding="utf-8")
    (out_dir / "phrases.txt").write_text("\n".join(phrases), encoding="utf-8")
    
    # Save top ranked images
    for rank, p in enumerate(ranked_paths, 1):
        shutil.copy(p, out_dir / f"top{rank}.jpg")
    
    print(f"✓ [{scene_idx:02d}] saved to {out_dir.relative_to(output_dir.parent)}")
    
    # Return row for CSV
    return [scene_dir.name] + [p.parent.name for p in ranked_paths]


def main():
    """Main function to run the scene-object matching pipeline."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Convert string device specs to torch.device objects
    internvl_device = torch.device(args.internvl_device)
    clip_device = torch.device(args.clip_device)
    
    # Convert paths to Path objects
    scenes_dir = Path(args.scenes_dir)
    objects_dir = Path(args.objects_dir) 
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print(f"Loading InternVL model from {args.internvl_model}...")
    internvl_model, internvl_tokenizer, generation_config = load_internvl_model(
        args.internvl_model, internvl_device)
    
    print(f"Loading CLIP model {args.clip_model}...")
    clip_model, clip_tokenizer, preprocess = load_clip_model(
        args.clip_model, args.clip_pretrained, clip_device)
    
    print(f"Loading spaCy model {args.spacy_model}...")
    nlp = load_spacy_model(args.spacy_model)
    
    # Encode all object images
    obj_paths, obj_feats = encode_all_objects(objects_dir, clip_model, preprocess, clip_device)
    print(f"Encoded {len(obj_paths)} object images")
    
    # Process all scenes
    print(f"Processing {len(list(scenes_dir.iterdir()))} scenes...")
    csv_rows = []
    
    for scene_dir in sorted(scenes_dir.iterdir()):
        if scene_dir.is_dir():
            row = process_scene(
                scene_dir=scene_dir,
                output_dir=output_dir,
                obj_paths=obj_paths,
                obj_feats=obj_feats,
                clip_model=clip_model,
                clip_tokenizer=clip_tokenizer,
                internvl_model=internvl_model,
                internvl_tokenizer=internvl_tokenizer,
                generation_config=generation_config,
                nlp=nlp,
                top_k=args.top_k,
                eval_k=args.eval_k,
                input_size=args.input_size,
                max_patches=args.max_patches,
                phrase_weight=args.phrase_weight,
                internvl_weight=args.internvl_weight,
                clip_device=clip_device,
                internvl_device=internvl_device
            )
            if row:
                csv_rows.append(row)
    
    # Save results to CSV
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        if csv_rows:
            writer.writerows(csv_rows)
            print(f"📄 CSV written to {csv_path}")
        else:
            print("⚠️ No results to write to CSV.")

if __name__ == "__main__":
    main()