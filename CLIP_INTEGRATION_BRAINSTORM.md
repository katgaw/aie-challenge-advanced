# CLIP Integration Brainstorming for Turkey Image Matching

## Overview
CLIP (Contrastive Language-Image Pre-training) is perfect for this use case because it embeds both images and text into the same semantic space, allowing direct similarity comparison between mood descriptions and turkey images.

## Approach 1: OpenAI CLIP (openai-clip) - Direct Implementation

### Pros:
- Official OpenAI implementation
- Well-documented
- Good performance
- Supports multiple model sizes (ViT-B/32, ViT-B/16, ViT-L/14, etc.)

### Cons:
- Requires PyTorch (larger dependency)
- Needs GPU for best performance (CPU works but slower)
- Model download on first use (~150MB-1GB depending on model)

### Implementation Structure:
```python
import clip
import torch
from PIL import Image
import faiss
import numpy as np

# Initialize CLIP model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Pre-embed all turkey images into FAISS index
# On each request: embed mood text, search FAISS, return best match
```

### Code Flow:
1. **Startup**: Load CLIP model, preprocess all turkey images, create embeddings, build FAISS index
2. **Request**: User provides mood → embed mood text → FAISS search → find best matching image → analyze with OpenAI vision API

---

## Approach 2: Sentence-Transformers CLIP (sentence-transformers)

### Pros:
- Easier to use (higher-level API)
- No PyTorch dependency (uses transformers)
- Can use different CLIP variants
- Good documentation

### Cons:
- Slightly less control
- May have different performance characteristics

### Implementation Structure:
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch

# Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# For images: model.encode(image)
# For text: model.encode(text)
# Both return same-dimensional embeddings
```

---

## Approach 3: Hugging Face Transformers CLIP

### Pros:
- Most flexible
- Access to many CLIP variants
- Can use different backends (PyTorch, TensorFlow)
- Community support

### Cons:
- More boilerplate code
- Need to handle preprocessing manually

### Implementation Structure:
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

---

## Approach 4: OpenAI CLIP API (if available)

### Pros:
- No local model loading
- No GPU/CPU concerns
- Always up-to-date

### Cons:
- API costs per request
- Network latency
- May not be available as a service

---

## Recommended Architecture

### Hybrid Approach: CLIP + FAISS + OpenAI Vision

```
┌─────────────────┐
│  User Request   │
│  (mood: "angry")│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CLIP Text      │
│  Embedding      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  FAISS Index    │─────▶│  Best Match:     │
│  (Image Embs)   │      │  "mad-turkey.png"│
└─────────────────┘      └────────┬─────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  OpenAI Vision   │
                         │  (Detailed      │
                         │   Analysis)      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  System Prompt   │
                         │  (with image    │
                         │   context)       │
                         └─────────────────┘
```

## Implementation Details

### 1. Model Selection
- **ViT-B/32**: Fastest, good quality (recommended for MVP)
- **ViT-B/16**: Better quality, slower
- **ViT-L/14**: Best quality, slowest

### 2. Embedding Strategy
- **Option A**: Embed images once at startup, store in FAISS
- **Option B**: Embed on-demand (slower but more flexible)
- **Recommendation**: Option A for production

### 3. Similarity Search
- **FAISS IndexFlatL2**: Simple, exact search (good for <10k images)
- **FAISS IndexIVFFlat**: Faster for larger datasets
- **Cosine similarity**: Can normalize embeddings and use dot product

### 4. Caching Strategy
- Cache image embeddings (persist FAISS index to disk)
- Cache mood→image mappings for common moods
- Lazy loading: only initialize CLIP when first needed

## Code Structure Proposal

```python
# api/index.py structure

# 1. Imports
import clip
import torch
from PIL import Image
import faiss
import numpy as np
from pathlib import Path

# 2. Global variables
clip_model = None
clip_preprocess = None
clip_device = None
image_vector_store = None
image_paths = []

# 3. Initialization function
def initialize_clip_vector_store():
    """Load CLIP model and create FAISS index of turkey images"""
    global clip_model, clip_preprocess, clip_device, image_vector_store, image_paths
    
    # Load CLIP
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)
    
    # Process all images
    # Build FAISS index
    # Store image paths

# 4. Search function
def find_similar_turkey_image(mood: str) -> str | None:
    """Use CLIP to find most similar turkey image to mood"""
    # Embed mood text
    # Search FAISS
    # Return best match path

# 5. Integration in chat endpoint
# Call find_similar_turkey_image() when mood is provided
```

## Performance Considerations

### Startup Time:
- CLIP model loading: ~2-5 seconds
- Image preprocessing: ~0.1-0.5s per image
- FAISS index creation: <1 second

### Request Time:
- Text embedding: ~10-50ms
- FAISS search: <1ms (for small dataset)
- Total: ~20-60ms per request

### Memory:
- CLIP model: ~300-500MB RAM
- Image embeddings: ~10 images × 512 dims × 4 bytes = ~20KB
- FAISS index: ~50KB

## Alternative: Lazy Loading Strategy

Instead of loading CLIP at startup, load it on first request:

```python
def get_clip_model():
    """Lazy load CLIP model"""
    global clip_model
    if clip_model is None:
        initialize_clip()
    return clip_model
```

## Error Handling Considerations

1. **Model download fails**: Fallback to filename matching
2. **Image not found**: Return None, continue without image context
3. **CLIP encoding fails**: Log error, use fallback
4. **FAISS search fails**: Return first image or None

## Testing Strategy

1. Test with various mood descriptions:
   - Exact matches: "annoyed" → "annoyed.png"
   - Semantic matches: "frustrated" → "annoyed.png"
   - Related concepts: "stressed" → "upset.png"
   - Unrelated: "happy" → should still find something reasonable

2. Performance benchmarks:
   - Startup time
   - Request latency
   - Memory usage

## Dependencies to Add

```toml
# pyproject.toml
dependencies = [
    "torch>=2.0.0",  # For CLIP
    "torchvision>=0.15.0",  # For image preprocessing
    "openai-clip>=1.0",  # Or use sentence-transformers
    "pillow>=10.0.0",  # For image loading
    "faiss-cpu>=1.7.4",  # For similarity search
    "numpy>=1.24.0",
]
```

## Next Steps

1. **Choose approach**: Recommend OpenAI CLIP (openai-clip) for best balance
2. **Implement initialization**: Load model and create FAISS index
3. **Implement search**: Text embedding + FAISS similarity search
4. **Integrate**: Add to chat endpoint
5. **Test**: Verify matching quality and performance
6. **Optimize**: Add caching, lazy loading if needed

