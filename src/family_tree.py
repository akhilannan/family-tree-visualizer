import json
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageSequenceClip, AudioFileClip

# Configuration
CONFIG_PATH = 'data/family.json'
MUSIC_PATH = 'assets/audio/background.mp3'
OUTPUT_PATH = 'family_video.mp4'
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 1600
HEADER_Y = 50
LEVEL_HEIGHT = 270
SPACER = 20
GROUP_PIC_MAX_WIDTH = 1000  # Maximum width for family photos
GROUP_PIC_MAX_HEIGHT = 600  # Maximum height for family photos
LABEL_FONT_SIZE = 24
LINE_SPACING = 40

# Image paths
PROFILE_IMAGES_PATH = 'assets/images/profiles'
FAMILY_IMAGES_PATH = 'assets/images/families'
DEFAULT_IMAGES_PATH = 'assets/images/defaults'

def preprocess_config(config):
    """
    Preprocess the configuration to ensure spouse gender is set appropriately.
    """
    def process_node(node):
        if 'spouse' in node and node['spouse']:
            spouse = node['spouse']
            if not spouse.get('gender'):
                spouse['gender'] = 'female' if node.get('gender', '').lower() == 'male' else 'male'
        for child in node.get('children', []):
            process_node(child)
    for root in config.get('root', []):
        process_node(root)
    return config

def get_image_path(name, image_type='profile'):
    """
    Retrieve the cached image path for a given name and image type.
    """
    name = name.strip().lower()
    if image_type == 'profile':
        return profile_files.get(name)
    else:
        return family_files.get(name)

def get_default_image(gender):
    """
    Retrieve the default image path for a given gender.
    """
    gender = gender.lower()
    if gender == 'male':
        return default_male
    elif gender == 'female':
        return default_female
    return None

def get_family_image(member):
    """
    Retrieve the family photo for a member, if available.
    """
    if member.get('hasFamily') is False:
        return None
        
    family_photo = get_image_path(member['name'], 'family')
    if family_photo:
        return family_photo
        
    if member.get('hasFamily') is True and default_family:
        return default_family
            
    return None

def has_family_photo(member):
    """
    Check if the member has a family photo.
    """
    return get_family_image(member) is not None

def get_tile(member, config, font, label=None):
    """
    Create a tile for a member with their profile image and label.
    Dynamically adjusts the tile width so the full label is always visible.
    """
    base_img_size = 200
    text_padding = 10  # extra padding around text

    base_name = member['name']
    display_name = f"{base_name} ({label})" if label else base_name

    # Create a temporary image to measure text size.
    temp_img = Image.new('RGB', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    text_bbox = temp_draw.textbbox((0, 0), display_name, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    # Determine tile width: at minimum base_img_size, or wider if label is longer.
    tile_width = max(base_img_size, text_width + 2 * text_padding)
    tile_height = base_img_size + 50  # extra space for label below image

    tile = Image.new('RGB', (tile_width, tile_height), 'white')

    # Attempt to open the profile image.
    img = None
    profile_path = get_image_path(base_name, 'profile')
    if profile_path:
        try:
            img = Image.open(profile_path)
        except Exception as e:
            print(f"Error opening image {profile_path}: {e}")

    if not img:
        default_path = get_default_image(member.get('gender', 'male'))
        if default_path:
            try:
                img = Image.open(default_path)
            except Exception as e:
                print(f"Error opening default image {default_path}: {e}")

    if not img:
        # If no image available, create a placeholder.
        color = 'dodgerblue' if member.get('gender', 'male').lower() == 'male' else 'hotpink'
        img = Image.new('RGB', (base_img_size, base_img_size), color)
        d = ImageDraw.Draw(img)
        d.text((base_img_size//2, base_img_size//2), base_name, fill='white', anchor='mm', font=font)
    else:
        img = img.resize((base_img_size, base_img_size))
    
    # Paste the image centered horizontally.
    img_x = (tile_width - base_img_size) // 2
    tile.paste(img, (img_x, 0))
    
    # Draw the label centered below the image.
    d = ImageDraw.Draw(tile)
    text_y = base_img_size + ( (tile_height - base_img_size) // 2 )
    d.text((tile_width // 2, text_y), display_name, fill='black', font=font, anchor='mm')
    
    return tile

def paste_tiles(canvas, tiles, y):
    """
    Paste tiles horizontally centered on the canvas.
    """
    total_width = sum(tile.width for tile in tiles) + SPACER * (len(tiles) - 1)
    start_x = (CANVAS_WIDTH - total_width) // 2
    for tile in tiles:
        canvas.paste(tile, (start_x, y))
        start_x += tile.width + SPACER

def draw_person(canvas, node, y, show_spouse, config, font, is_root=False, label=None):
    """
    Draw a person (and their spouse, if applicable) on the canvas.
    """
    tiles = []
    if show_spouse and node.get('spouse'):
        main_label = label if label is not None else (None if is_root else "Child")
        tiles.append(get_tile(node, config, font, label=main_label))
        tiles.append(get_tile(node['spouse'], config, font, label="Spouse"))
    else:
        tiles.append(get_tile(node, config, font, label=label))
    
    paste_tiles(canvas, tiles, y)

def draw_ancestors(canvas, ancestors, header_y, level_height, config, font):
    """
    Draw ancestors on the canvas.
    """
    for i, (node, lbl) in enumerate(ancestors):
        is_root = (i == 0)
        draw_person(canvas, node, header_y + i * level_height, True, config, font, is_root, label=lbl)

def draw_centered_text(draw, y, text, font):
    """
    Draw centered text on the canvas.
    """
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text(((CANVAS_WIDTH - text_width) // 2, y), text, fill="black", font=font)

def create_focused_slideshow(config, slides_folder, font, label_font):
    """
    Create a slideshow focusing on family members and their relationships.
    """
    os.makedirs(slides_folder, exist_ok=True)
    slide_files = []

    root = config['root'][0]
    print("Generating slide 0: Root only")
    canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
    draw_person(canvas, root, HEADER_Y, True, config, font, is_root=True)
    slide_path = os.path.join(slides_folder, "slide_focused_0.jpg")
    canvas.save(slide_path)
    slide_files.append(slide_path)
    
    print("Generating slide 1: Root with children")
    canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
    draw_person(canvas, root, HEADER_Y, True, config, font, is_root=True)
    children = root.get('children', [])
    if children:
        child_tiles = [get_tile(child, config, font, label=f"Child {i+1}") for i, child in enumerate(children)]
        paste_tiles(canvas, child_tiles, HEADER_Y + LEVEL_HEIGHT)
    slide_path = os.path.join(slides_folder, "slide_focused_1.jpg")
    canvas.save(slide_path)
    slide_files.append(slide_path)

    slide_index = 2

    def recursive_focus(ancestors, focus_node, slide_index, label=None):
        """
        Recursively generate slides focusing on descendants.
        Updated:
          - Always generate a base focus slide for the node.
          - If children exist, always generate a group slide showing the children.
          - For each child: if the child has a family pic, generate a special family slide (with centered multiple
            label lines) and skip further recursion for that child. Otherwise, recurse normally.
        """
        # Base slide: show the focused node normally.
        print(f"Generating base slide {slide_index} for {focus_node['name']}")
        canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
        draw_ancestors(canvas, ancestors, HEADER_Y, LEVEL_HEIGHT, config, font)
        draw_person(canvas, focus_node, HEADER_Y + len(ancestors)*LEVEL_HEIGHT, True, config, font, label=label)
        slide_path = os.path.join(slides_folder, f"slide_focused_{slide_index}.jpg")
        canvas.save(slide_path)
        slide_files.append(slide_path)
        slide_index += 1

        children = focus_node.get('children', [])
        if children:
            # Group slide: show the focused node again with its children tiled below.
            print(f"Generating group slide {slide_index} for children of {focus_node['name']}")
            canvas2 = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
            draw_ancestors(canvas2, ancestors, HEADER_Y, LEVEL_HEIGHT, config, font)
            draw_person(canvas2, focus_node, HEADER_Y + len(ancestors)*LEVEL_HEIGHT, True, config, font, label=label)
            child_tiles = [get_tile(child, config, font, label=f"Child {i+1}") for i, child in enumerate(children)]
            paste_tiles(canvas2, child_tiles, HEADER_Y + (len(ancestors)+1)*LEVEL_HEIGHT)
            slide_path = os.path.join(slides_folder, f"slide_focused_{slide_index}.jpg")
            canvas2.save(slide_path)
            slide_files.append(slide_path)
            slide_index += 1

            # Recurse for each child.
            new_ancestors = ancestors + [(focus_node, label)]
            for i, child in enumerate(children):
                child_label = f"Child {i+1}"
                # If a child has a family pic, show the special family slide and skip further recursion.
                if has_family_photo(child):
                    print(f"Generating family slide {slide_index} for {child['name']}")
                    canvas3 = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), 'white')
                    draw_ancestors(canvas3, new_ancestors, HEADER_Y, LEVEL_HEIGHT, config, font)
                    # Process the family image.
                    family_photo_path = get_family_image(child)
                    try:
                        group_img = Image.open(family_photo_path)
                        # Calculate new dimensions while preserving aspect ratio.
                        aspect_ratio = group_img.width / group_img.height
                        if aspect_ratio > GROUP_PIC_MAX_WIDTH / GROUP_PIC_MAX_HEIGHT:
                            new_width = GROUP_PIC_MAX_WIDTH
                            new_height = int(new_width / aspect_ratio)
                        else:
                            new_height = GROUP_PIC_MAX_HEIGHT
                            new_width = int(new_height * aspect_ratio)
                        group_img = group_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    except Exception as e:
                        print(f"Error opening family photo {family_photo_path}: {e}")
                        group_img = Image.new('RGB', (GROUP_PIC_MAX_WIDTH, GROUP_PIC_MAX_HEIGHT), 'gray')
                    # Center the family photo.
                    x = (CANVAS_WIDTH - group_img.width) // 2
                    y_group = HEADER_Y + len(new_ancestors) * LEVEL_HEIGHT
                    canvas3.paste(group_img, (x, y_group))
                    # Draw centered multi-line labels.
                    draw_obj = ImageDraw.Draw(canvas3)
                    y_text = y_group + group_img.height + LINE_SPACING
                    # Line 1: the focused child and its spouse (if any).
                    line1 = f"{child['name']} (Child {i+1})"
                    if child.get('spouse'):
                        spouse_name = child['spouse'].get('name', '')
                        if spouse_name:
                            line1 += f", {spouse_name} (Spouse)"
                    draw_centered_text(draw_obj, y_text, line1, label_font)
                    y_text += LABEL_FONT_SIZE + LINE_SPACING
                    # Line 2: labels for the children included in the group picture.
                    if child.get('children'):
                        child_labels = [
                            f"{c['name']} (Child {j+1})" for j, c in enumerate(child['children'])
                        ]
                        line2 = ", ".join(child_labels)
                        draw_centered_text(draw_obj, y_text, line2, label_font)
                        y_text += LABEL_FONT_SIZE + LINE_SPACING
                    slide_path = os.path.join(slides_folder, f"slide_focused_{slide_index}.jpg")
                    canvas3.save(slide_path)
                    slide_files.append(slide_path)
                    slide_index += 1
                    # Skip further recursion for this child.
                else:
                    slide_index = recursive_focus(new_ancestors, child, slide_index, label=child_label)
        return slide_index

    for i, child in enumerate(children):
        if child.get('spouse') or child.get('children'):
            slide_index = recursive_focus([(root, None)], child, slide_index, label=f"Child {i+1}")
    
    return slide_files

def main():
    """
    Main function to generate the family video.
    """
    slides_folder = 'slides'
    os.makedirs(slides_folder, exist_ok=True)

    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {CONFIG_PATH} not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        exit(1)

    config = preprocess_config(config)

    global profile_files, family_files
    profile_files = {
        os.path.splitext(f)[0].lower(): os.path.join(PROFILE_IMAGES_PATH, f)
        for f in os.listdir(PROFILE_IMAGES_PATH)
        if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    }
    family_files = {
        os.path.splitext(f)[0].lower(): os.path.join(FAMILY_IMAGES_PATH, f)
        for f in os.listdir(FAMILY_IMAGES_PATH)
        if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    }

    try:
        font = ImageFont.truetype("arial.ttf", 18)
        label_font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
    except IOError:
        font = ImageFont.load_default()
        label_font = font  # Use default for labels as well

    global default_male, default_female, default_family
    default_male = os.path.join(DEFAULT_IMAGES_PATH, 'male.jpg')
    default_female = os.path.join(DEFAULT_IMAGES_PATH, 'female.jpg')
    default_family = os.path.join(DEFAULT_IMAGES_PATH, 'family.jpg')

    if not os.path.exists(default_male):
        default_male = None
    if not os.path.exists(default_female):
        default_female = None
    if not os.path.exists(default_family):
        default_family = None

    slide_files = create_focused_slideshow(config, slides_folder, font, label_font)
    
    clip = ImageSequenceClip(slide_files, fps=1/5)
    audio = AudioFileClip(MUSIC_PATH).with_duration(clip.duration)
    final_clip = clip.with_audio(audio)
    final_clip.write_videofile(OUTPUT_PATH, codec='libx264', audio_codec='aac')

    shutil.rmtree(slides_folder)

if __name__ == '__main__':
    main()