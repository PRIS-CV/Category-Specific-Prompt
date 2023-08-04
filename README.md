# Category-Specific-Prompt
Code release for "Category-Specific Prompts for Animal Action Recognition with Pretrained Vision-Language Models" (ACM MM 23)

Animal action recognition has a wide range of applications. However, the task largely remains unexplored due to the greater challenges compared to human action recognition, such as lack of annotated training data, large intra-class variation caused by diverse animal morphology, and interference of cluttered background in animal videos. Most of the existing methods directly apply human action recognition techniques, which essentially require a large amount of annotated data. In recent years, contrastive vision-language pretraining has demonstrated strong few-shot generalization ability and has been used for human action recognition. Inspired by the success, we develop a highly performant action recognition framework based on the CLIP model. Our model addresses the above challenges via a novel category-specific prompt adaptation module to generate adaptive prompts for both text and video based on the animal category detected in input videos. On one hand, it can generate more precise and customized textual descriptions for each action and animal category pair, being helpful in the alignment of textual and visual space. On the other hand, it allows the model to focus on video features of the target animal in the video and reduce the interference of video background noise. Experimental results demonstrate that our method outperforms five previous behavior recognition methods on the Animal Kingdom dataset and has shown best generalization ability on unseen animals.

**Model structure:**
![](https://github.com/jynkris1016/Category-Specific-Prompt/blob/main/img/Model%20structure.png)
**Some prediction results:**
![](https://github.com/jynkris1016/Category-Specific-Prompt/blob/main/img/visualization.png)

## Requirements
`<pip install -r requirements.txt>`

## Train
`<python -m torch.distributed.launch --nproc_per_node=<YOUR_NPROC_PER_NODE> main.py -cfg <YOUR_CONFIG> --output <YOUR_OUTPUT_PATH> --accumulation-steps 4>`

## Test
`<python -m torch.distributed.launch --nproc_per_node=<YOUR_NPROC_PER_NODE> main.py -cfg <YOUR_CONFIG> --output <YOUR_OUTPUT_PATH> --only_test --opts TEST.NUM_CLIP 4 TEST.NUM_CROP 3 --resume <YOUR_MODEL_FILE>>`
