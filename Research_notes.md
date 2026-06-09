# Transferable Model-agnostic Vision-Language Model Adaptation for Efficient Weak-to-Strong Generalization

## Abstract
Transferable Model-agnostic adapter (TransMiter) improves VLMs without backpropagation. TransMiter captures the knowledge gap between pre-trained and fine-tuned VLMs, in an unsupervised manner. TransMiter consists of only a few layers inducing a negligible additional inference cost. supplementing the process with a few labeled data further yields additional performance gain, often surpassing a fine-tuned stronger model, with a marginal training cost.

## Introduction
due to semantic discrepancy or dimensional mismatch between different models, directly reusing a subset of parameters from a weaker model to a stronger one. Requires additional training to align their internal representations properly.

Some works mitigate this issue by enforcing semantic consistency (e.g., model prediction), reducing the need for re-training but at the cost of slower inference due to multiple model usage. Alternatively, knowledge distillation has been demonstrated as an effective way for smaller models to guide larger ones. Still, this approach requires expensive retraining whenever a new model is introduced, limiting its practicality for rapidly evolving models.

* **Model-agnostic compatibility:** It should plug into any new model
* **Computationally efficient transfer:** It must transfer the knowledge fast and cheap—specifically "without backpropagation" 
* **Minimal additional inference cost:** It won't slow the AI down when a user is actually using it. 

First, the authors take their standard, out-of-the-box Strong Model, and compare it to their custom-trained Weak Model. They feed them an image and look at the differences in their logits (final answers). By looking at the difference, TransMiter calculates the knowledge gap between a normal model and a smart, specialized model. It does this in an unsupervised way. Because every single Vision-Language Model outputs final answers in the exact same format. TransMiter can take that "knowledge gap" it calculated from the weak model and simply mathematically apply it directly onto the brand-new Strong Model's final answers.

* **Auxiliary Classes**—extra, random background categories to the logits that aren't the main target. By increasing the dimensionality of logit vectors, you give the model more room to express subtle hints. 
* **Basis change:** master algebra equation (a closed-form solution) where you just plug in the numbers and the exact answer come out instantly. 

TransMiter becomes a strong initialization point as a small amount of labeled data can help boost performance, that it may surpass fine tuned models. TransMiter has a very lightweight design as it only contains of simple projection matrices and a single MLP layer. 

## Related Works
Existing methods neglect the rapid evolution in the size and complexity of models which require retraining.
Previous Methods try to transfer a portion of the model’s internal parameters or the complete model from a weaker model to a stronger one but the architectural differences hinder it or the retraining is expensive. 

## Preliminaries
* **VLM:** image x passed through pretrained VLM, cosine similarity is performed on the encoded features by the image encoder and the encoded features of each task class given by the text encoder. The output is the class with the highest cosine similarity.
* **TransMiter:** A knowledge extraction operation is performed on the pretrained model and its fine-tuned counterpart to get the knowledge gap, which is then transferred to the pretrained target model using a transfer operation resulting in a knowledge enhanced target model whose performance should theoretically equal or better be than the better model among the finetuned source model and the pretrained target model and worse or equal than the finetuned version of the target model.

## Method
TransMiter uses the logits from the VLM as inputs as The logits exhibit two key properties: 
(1) a fixed dimensionality that aligns with the number of classes, 
(2) semantic consistency for each logit element.

It achieves high efficiency as it does not require access to the parameters of the VLMs.

The logits from the weak pretrained source VLM is projected into a D Dimensional space by multiplying with the projection matrix Win producing the input feature hs which is passed through the MLP Layer. ( hs hat  = hs + MLP(hs) ) which is finally projected back into the original logit space by multiplying with the reconstruction matrix Wout.
Win is equal to the transpose of Wout. This ensures that the generalized knowledge of the model is preserved. Since they are orthogonal matrices, Win.Wout = I. Wout is now assumed to be Ws.

Loss Function L = KL(softmax(zft-s/temperature coefficient of ft-s),softmax(z hat s/temperature coefficient of pt-s))

Auxiliary classes’ candidates are collected from OpenImages. Therefore the transition matrix Ws is reshaped so that output logits have M(Naux + Ntask) dimensionality.

straight forward transfer: take pretrained strong VLM’s logits and pass it through the same formula with the weak model’s normal matrix but discrepancies between logits for strong and the weak VLM will lead to suboptimal transferability.

We feed some unlabeled images to both the strong and the weak VLMs. We get the input features of both by multiplying the obtained logits with transpose of Ws. We then find a mapping between the input features of both VLMs. We aim to minimise the distance between hs and ht of all the images with a regularization term proportional to distance between W and the identity matrix. We can obtain the solution to this problem using SVD making backpropogation unnecessary. Therefore projection matrix for strong VLM becomes W s T. W hat and similarly reconstruction matrix becomes W hat T. Ws

## Experiments
* TransMiter adds only 0.01 GFLOPs while EFT takes 12 to 36 FLOPs
* In some cases where zero shot strong VLM was quite better than fine tuned weak VLM, transMiter holds back the strong model forcing it to act like the weak model in cases of common topics such as food. 
* TransMiter makes fine tuning methods also better. It not only improves the scores but also preserves it’s FPS speed for all methods.