# PDQ Hash

This hash is a pure Rust implementation of the [PDQ algorithm](https://github.com/facebook/ThreatExchange/tree/main/pdq) described [here](https://raw.githubusercontent.com/facebook/ThreatExchange/main/hashing/hashing.pdf).

## Calculation
The PDQ algorithm was developed and open-sourced by Facebook (now Meta) in 2019. It specifies a transformation which converts images into a binary format ('PDQ Hash') whereby 'perceptually similar’ images produce similar outputs. It was designed to offer an industry standard for representing images to collaborate on threat mitigation.

![Obtain PDQ Hash of an image](/docs/transformation.png)

Comparing two images reduces to computing distance (for example, Hamming distance) between their representations, or as % bit similarity. 
![Comparing two bit strings](/docs/comparison.png)

(16 bits are just used here for easier interpretation; PDQ hashes represent 256 bits)
## Consider additional image transformations

Additionally, PDQ hashes for rotations and mirrors of the original image can be inferred efficiently, by just manipulating the Discrete Cosine Transform created in latter stages of processing.
![Comparing two bit strings](/docs/rotation.png)
Example: PDQ Hash of mirrored original image only requires manipulation of the transform

DCT Manipulation needed for corresponding PDQ Hash
* Mirrored Y 
** Negate alternate columns
* Mirrored X
** Negate alternate rows
* Mirrored Main Diagonal
** Transpose
* Mirrored Off Diagonal
** Negate off-diagonal, transpose
* Rotated 90
** Negate alternate columns, transpose
* Rotated 180
** Negate off-diagonal
* Rotated 270
** Negate alternate rows, transpose

## Offering similarity resilience

The resulting hashes are resilient to certain transformations, some more so than others, to detect additional attempted manipulation. Generally, images retaining overall structure are more resilient than changes to pixel positions and larger areas of pixel change. 
![Obtain PDQ Hash of an image](/docs/docs/similarity.png)

Transformations that result in similar hashes:
* File format change
* Quality reduction
* Light crops and shifts
* Rotations (when additional hashes compared)
* Resizing
* Light watermarks
* Mirroring (when additional hashes compared)
* Noise or filter applied
* Light logos
