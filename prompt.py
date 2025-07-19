t2i_long_text_prompt_old = """
You're a prompt generator for text-to-image models.

1. Analyze the provided image description to determine the best task type:
   - Type A: Accurate count of multiple objects (focuses on enumerating objects without extra details)
   - Type B: Multiple objects + attributes (color, material, text)
   - Type C: Multiple objects + absolute positions (e.g., "left", "center", using spatial references)
   - Type D: Multiple objects + scene description (e.g., setting, lighting, background context)
   After reasoning, select only ONE task type.

2. Generate prompt: Based on the task type, create a detailed English prompt with rules:
   - Use English ONLY.
   - Include exact counts (e.g., "three apples", not "several apples").
   - Ensure the prompt token length is between 100 to 256 tokens (aim for 80-200 words). Provide rich details like attributes, positions, or scene elements as needed.
   - Focus on clarity and specificity for high-quality image generation.

3. Output format: Your final response MUST be structured as:
   - Start with a reasoning section (your step-by-step thought process).
   - Then output <task>selected_letter</task> (e.g., <task>B</task>).
   - Finally output <prompt>generated_prompt</prompt>, where generated_prompt is the full text prompt.

Some description of this image:
{description}

Example:
Reasoning: The image description mentions "five coffee mugs with different brand logos, made of red ceramic, placed on a wooden table in a sunny kitchen". This involves multiple objects (mugs) combined with specific attributes (color: red, material: ceramic, text: logos) and some scene elements. Task type B is the best fit because attributes are emphasized, while scene details are secondary. Type A lacks attributes, Type C doesn't need positions, and Type D's scene focus is not primary. For the prompt, I will ensure exact counts (five mugs), vivid attribute descriptions, and minimal scene context to stay within token limits.
<task>B</task>
<answer>In a brightly lit kitchen with large windows allowing natural sunlight to flood in, five identical red ceramic coffee mugs are neatly arranged in a straight line on a rustic wooden table. Each mug features a distinct corporate logo printed prominently in glossy white: the first mug displays 'Starbucks', the second shows 'Nespresso', the third has 'Lavazza', the fourth is branded 'Illy', and the fifth exhibits 'Peet's Coffee'. The ceramic surface is smooth and reflective, highlighting the vibrant red hue under the warm lighting. The logos are positioned centrally on each mug, with crisp white text contrasting against the rich red background. Wood grain textures on the table add a natural element, while a hint of steam rises from one mug, suggesting freshly brewed coffee. Overall, the scene balances brand diversity with precision in object placement and material details, creating a clean, photorealistic composition.</answer>
""".strip()


t2i_long_text_prompt = """
You're a prompt generator for text-to-image models. Perform these steps:

1. **Think step by step**: Analyze the image description to choose the best task type:
   A. Accurate count of multiple objects (focuses on enumerating objects without extra details)
   B. Multiple objects + attributes (color, material, text)
   C. Multiple objects + absolute positions (e.g., "left", "center", using spatial references)
   D. Multiple objects + scene description (e.g., setting, lighting, background context)
   Select only ONE type.

2. **Generate prompt**:
   - English ONLY
   - Include exact counts
   - Ensure token length is 100-256 tokens (approx 60-120 words)
   - Provide rich details (attributes, positions, scene)

3. **Output format**:
   - Start with a brief reasoning
   - Then: <task>selected_letter</task>
   - Finally: <answer>generated_prompt</answer>
   - ALL generated prompt MUST appear between <answer> and </answer>

Image description:
{description}

Example:
"
The description mentions 'three red apples and two green pears on a wooden table' - objects with attributes (color) take priority over positions or scene. Task B fits best.
<task>B</task>
<answer>
Three red apples with visible stems and smooth skin textures sit beside two matte green pears arranged diagonally across a rustic oak table surface. Each apple has distinct speckling patterns transitioning from ruby to crimson, while the pears feature subtle yellow patches at their bases.</answer>
"
"""


# 图像编辑任务的请求Prompt模板
edit_long_text_prompt = """
<image>
You are an AI image editor prompt generator. Analyze the current image and its description to:

1. **Think step by step**: 
   - Identify key elements in the image that need editing
   - Determine the most suitable task type:
      A. Object management (add/remove objects)
      B. Attribute editing (modify properties like color/texture)
      C. Scene transformation (change background/style)
   - Select ONE best-fitting type

2. **Generate editing instructions**:
   - Create THREE parallel, non-conflicting English instructions 
   - Each instruction must be exactly one complete sentence without numbering or bullet points
   - Start each sentence with imperative verbs: "Add", "Remove", "Change", "Make"
   - Include specific object references where needed
   - Total word count under 40 words
   - Each instruction must be fully independent and executable alone

3. **Output format**:
   - First, provide brief reasoning
   - Then output: <task>selected_letter</task>
   - Finally output: <answer>one instruction per line</answer>
   - ALL editing instructions must appear between <answer> and </answer>
   - Use simple line breaks between instructions (no numbering)

Current image description:
{description}

Example:
Reasoning: Beach scene with empty foreground needs object additions. Task type A fits best to introduce sea life elements without altering existing attributes.
<task>A</task>
<answer>
Add a starfish near the left rocks.
Add two seagulls above the waves.
Add a red sailboat on the horizon.
</answer>
""".strip()

t2i_complex_prompt = """
You're a text-to-image prompt generator. Create an English prompt under 100 tokens including:
1. Exact object counts (e.g., "three apples")
2. Key attributes like colors/textures (e.g., "red glossy")
3. Spatial relationships (e.g., "on the left", "stacked on top")

Focus on concise yet descriptive details.

**Output format**:
   - Start with a brief reasoning
   - Finally: <answer>generated_prompt</answer>
   - ALL generated prompt MUST appear between <answer> and </answer>

Image description:
{description}

Example:
Description specifies two objects with colors and positions
<answer>Two dogs: a golden retriever on the left sitting beside a black poodle on the right, grassy park background, sunny day.</answer>
""".strip()

edit_complex_prompt = """
You're an advanced image editing instruction generator.
Analyze implicit semantics to create ONLY ONE English instruction sentence under 30 words:
1. Object modifications ("add two", "remove the")
2. Key attribute changes ("change to blue", "make metallic")
3. Position adjustments ("move left", "swap positions")

**Output format**:
   - First: Start with a brief reasoning about what should be edited
   - Final: <answer>edit_instruction</answer>
   - ALL generated prompt MUST appear between <answer> and </answer>

Image description:
{description}

Example:
Original description: "A red car parked beside a bicycle"
<answer>Change the car color to matte navy blue.</answer>
""".strip()