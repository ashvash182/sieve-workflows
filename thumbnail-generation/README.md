# thumbnail generation workflow
- takes in a video path, prompt, number of desired outputs, and optional title text
- uses pyscenedetect, CLIP image embeddings, and semantic search to locate keyframes closest to thumbnail prompt
- (in progress) overlays title text (if provided, or generates from video contents) using google fonts API