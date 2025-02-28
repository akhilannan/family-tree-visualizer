import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageSequenceClip, AudioFileClip

# Configuration
CONFIG_PATH = "data/family.json"
MUSIC_PATH = "assets/audio/background.mp3"
OUTPUT_PATH = "family_video.mp4"
TREE_IMAGE_PATH = "complete_family_tree.jpg"
CANVAS_WIDTH = 1600
CANVAS_HEIGHT = 1600
HEADER_Y = 50
LEVEL_HEIGHT = 300
SPACER = 20
GROUP_PIC_MAX_WIDTH = 1000
GROUP_PIC_MAX_HEIGHT = 600
LABEL_FONT_SIZE = 24
LINE_SPACING = 40
LINE_MARGIN = 5
H_SPACER = 30
V_SPACER = 50
LINE_COLOR = "#D3D3D3"
LINE_WIDTH = 8

# Image paths
PROFILE_IMAGES_PATH = "assets/images/profiles"
FAMILY_IMAGES_PATH = "assets/images/families"
DEFAULT_IMAGES_PATH = "assets/images/defaults"


@dataclass
class ImageCache:
    """Class to store image file caches."""
    profile_files: Dict[str, str] = field(default_factory=dict)
    family_files: Dict[str, str] = field(default_factory=dict)
    default_male: Optional[str] = None
    default_female: Optional[str] = None
    default_family: Optional[str] = None

    def initialize(self):
        """Initialize the image caches."""
        allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}

        def load_image_dict(directory):
            return {
                os.path.splitext(f)[0].lower(): os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.splitext(f)[1].lower() in allowed_extensions
            }

        self.profile_files = load_image_dict(PROFILE_IMAGES_PATH)
        self.family_files = load_image_dict(FAMILY_IMAGES_PATH)

        # Set default images
        for image_type, filename in [
            ("default_male", "male.jpg"),
            ("default_female", "female.jpg"),
            ("default_family", "family.jpg")
        ]:
            path = os.path.join(DEFAULT_IMAGES_PATH, filename)
            setattr(self, image_type, path if os.path.exists(path) else None)


def preprocess_config(config: dict) -> dict:
    """Preprocess configuration to ensure spouse gender is set appropriately."""
    def process_node(node: dict) -> None:
        if "spouse" in node and node["spouse"]:
            spouse = node["spouse"]
            if not spouse.get("gender"):
                spouse["gender"] = "female" if node.get("gender", "").lower() == "male" else "male"
        for child in node.get("children", []):
            process_node(child)

    for root in config.get("root", []):
        process_node(root)
    return config


def get_image_path(name: str, image_type: str, cache: ImageCache) -> Optional[str]:
    """Retrieve the cached image path for a given name and image type."""
    key = name.strip().lower()
    return cache.profile_files.get(key) if image_type == "profile" else cache.family_files.get(key)


def get_default_image(gender: str, cache: ImageCache) -> Optional[str]:
    """Retrieve the default image path for a given gender."""
    return cache.default_male if gender.lower() == "male" else cache.default_female


def get_family_image(member: dict, cache: ImageCache) -> Optional[str]:
    """Retrieve the family photo for a member, if available."""
    if member.get("hasFamily") is False:
        return None

    family_photo = get_image_path(member["name"], "family", cache)
    return family_photo or (cache.default_family if member.get("hasFamily") else None)


def has_family_photo(member: dict, cache: ImageCache) -> bool:
    """Check if the member has a family photo."""
    return get_family_image(member, cache) is not None


def resize_image(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Resize image preserving aspect ratio to fit within max dimensions."""
    aspect_ratio = img.width / img.height
    if aspect_ratio > max_width / max_height:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def load_image(path: str, default_path: str = None, default_size: Tuple[int, int] = None) -> Optional[Image.Image]:
    """Load an image from the given path, falling back to default if needed."""
    try:
        return Image.open(path)
    except Exception as e:
        print(f"Error opening image {path}: {e}")
        if default_path:
            try:
                return Image.open(default_path)
            except Exception as e:
                print(f"Error opening default image {default_path}: {e}")
        if default_size:
            return Image.new("RGB", default_size, "gray")
    return None


def get_tile(node: dict, font: ImageFont.FreeTypeFont, cache: ImageCache, label: str = None) -> Image.Image:
    """Create a tile for a single person."""
    base_img_size = 200
    text_padding = 10
    shadow_offset = 4
    border_padding = 10
    corner_radius = 10
    base_name = node["name"]
    display_name = f"{base_name} ({label})" if label else base_name

    # Measure text width
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    text_bbox = temp_draw.textbbox((0, 0), display_name, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    # Calculate tile dimensions
    tile_width = max(base_img_size, text_width) + 2 * (border_padding + text_padding)
    tile_height = base_img_size + 50 + 2 * border_padding

    # Create canvas with shadow
    canvas = Image.new("RGBA", (tile_width + shadow_offset, tile_height + shadow_offset), (255, 255, 255, 0))
    shadow = Image.new("RGBA", (tile_width, tile_height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle([(0, 0), (tile_width - 1, tile_height - 1)], fill="#D3D3D3", radius=corner_radius)
    canvas.paste(shadow, (shadow_offset, shadow_offset), shadow)

    # Create main tile
    tile = Image.new("RGBA", (tile_width, tile_height), (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile)
    tile_draw.rounded_rectangle([(0, 0), (tile_width - 1, tile_height - 1)], fill="white", outline="#E8E8E8", radius=corner_radius, width=1)

    # Load profile image
    profile_path = get_image_path(base_name, "profile", cache)
    img = load_image(profile_path) if profile_path else None

    if not img:
        default_path = get_default_image(node.get("gender", "male"), cache)
        img = load_image(default_path) if default_path else None

    if img:
        aspect_ratio = img.width / img.height
        new_width = base_img_size if aspect_ratio > 1 else int(base_img_size * aspect_ratio)
        new_height = int(base_img_size / aspect_ratio) if aspect_ratio > 1 else base_img_size
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_x = (base_img_size - new_width) // 2 + (tile_width - base_img_size) // 2
        img_y = (base_img_size - new_height) // 2 + border_padding
        tile_draw.rectangle([((tile_width - base_img_size) // 2, border_padding),
                            ((tile_width + base_img_size) // 2, border_padding + base_img_size)], fill="white")
        tile.paste(img, (img_x, img_y))
    else:
        color = "dodgerblue" if node.get("gender", "male").lower() == "male" else "hotpink"
        img = Image.new("RGB", (base_img_size, base_img_size), color)
        d = ImageDraw.Draw(img)
        d.text((base_img_size // 2, base_img_size // 2), base_name, fill="white", anchor="mm", font=font)
        tile.paste(img, ((tile_width - base_img_size) // 2, border_padding))

    # Add text
    text_y = base_img_size + border_padding + ((tile_height - base_img_size - 2 * border_padding) // 2)
    tile_draw.text((tile_width // 2, text_y), display_name, fill="black", font=font, anchor="mm")
    canvas.paste(tile, (0, 0), tile)
    return canvas


def get_member_tile(member: dict, font: ImageFont.FreeTypeFont, cache: ImageCache, label: str = None) -> Image.Image:
    """Create a combined tile for a member and spouse if applicable."""
    if member.get("spouse"):
        tile1 = get_tile(member, font, cache, label=label)
        tile2 = get_tile(member["spouse"], font, cache, label="Spouse")
        height = max(tile1.height, tile2.height)
        width = tile1.width + SPACER + tile2.width
        combined = Image.new("RGB", (width, height), "white")
        combined.paste(tile1, (0, (height - tile1.height) // 2))
        combined.paste(tile2, (tile1.width + SPACER, (height - tile2.height) // 2))
        return combined
    return get_tile(member, font, cache, label=label)


def paste_tiles(canvas: Image.Image, tiles: list, y: int) -> list:
    """Paste tiles horizontally centered on the canvas."""
    total_width = sum(tile.width for tile in tiles) + SPACER * (len(tiles) - 1)
    start_x = (CANVAS_WIDTH - total_width) // 2
    positions = []
    for tile in tiles:
        canvas.paste(tile, (start_x, y))
        positions.append((start_x, y, tile.width, tile.height))
        start_x += tile.width + SPACER
    return positions


def draw_person(canvas: Image.Image, node: dict, y: int, show_spouse: bool, font: ImageFont.FreeTypeFont, 
                cache: ImageCache, is_root: bool = False, label: str = None) -> tuple:
    """Draw a person (and spouse) on the canvas and return connection point."""
    main_label = label or node.get("customLabel") or (None if is_root else "Child")
    
    if show_spouse and node.get("spouse"):
        tiles = [get_tile(node, font, cache, label=main_label), 
                get_tile(node["spouse"], font, cache, label="Spouse")]
        positions = paste_tiles(canvas, tiles, y)
        draw = ImageDraw.Draw(canvas)
        spouse_line_y = y + tiles[0].height // 2
        draw.line([(positions[0][0] + positions[0][2] + LINE_MARGIN, spouse_line_y),
                  (positions[1][0] - LINE_MARGIN, spouse_line_y)], fill=LINE_COLOR, width=LINE_WIDTH)
    else:
        tiles = [get_tile(node, font, cache, label=main_label)]
        positions = paste_tiles(canvas, tiles, y)

    left = positions[0][0]
    right = positions[-1][0] + positions[-1][2]
    mid_x = (left + right) // 2
    bottom_y = max(pos[1] + pos[3] for pos in positions)
    return (mid_x, bottom_y + LINE_MARGIN)


def draw_ancestors(canvas: Image.Image, ancestors: list, header_y: int, level_height: int, 
                  font: ImageFont.FreeTypeFont, cache: ImageCache) -> None:
    """Draw ancestors on the canvas."""
    for i, (node, lbl) in enumerate(ancestors):
        draw_person(canvas, node, header_y + i * level_height, True, font, cache, is_root=(i == 0), label=lbl)


def draw_centered_text(draw: ImageDraw.ImageDraw, y: int, text: str, font: ImageFont.FreeTypeFont) -> None:
    """Draw centered text on the canvas."""
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text(((CANVAS_WIDTH - text_width) // 2, y), text, fill="black", font=font)


def create_focused_slideshow(config: dict, slides_folder: str, font: ImageFont.FreeTypeFont, 
                            label_font: ImageFont.FreeTypeFont, cache: ImageCache) -> list:
    """Create a slideshow focusing on family members and relationships."""
    os.makedirs(slides_folder, exist_ok=True)
    slide_files = []
    root = config["root"][0]

    # Generate root slide
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw_person(canvas, root, HEADER_Y, True, font, cache, is_root=True)
    slide_path = os.path.join(slides_folder, "slide_focused_0.jpg")
    canvas.save(slide_path)
    slide_files.append(slide_path)

    # Generate root with children slide
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw_person(canvas, root, HEADER_Y, True, font, cache, is_root=True)
    children = root.get("children", [])
    if children:
        child_tiles = [get_tile(child, font, cache, label=child.get("customLabel", f"Child {i+1}")) 
                      for i, child in enumerate(children)]
        paste_tiles(canvas, child_tiles, HEADER_Y + LEVEL_HEIGHT)
    slide_path = os.path.join(slides_folder, "slide_focused_1.jpg")
    canvas.save(slide_path)
    slide_files.append(slide_path)

    slide_index = 2

    def recursive_focus(ancestors: list, focus_node: dict, slide_index: int, label: str = None) -> int:
        """Recursively generate slides for family members."""
        print(f"Generating base slide {slide_index} for {focus_node['name']}")
        base_canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
        draw_ancestors(base_canvas, ancestors, HEADER_Y, LEVEL_HEIGHT, font, cache)
        parent_conn = draw_person(base_canvas, focus_node, HEADER_Y + len(ancestors) * LEVEL_HEIGHT, 
                                 True, font, cache, label=label)
        slide_path = os.path.join(slides_folder, f"slide_focused_{slide_index}.jpg")
        base_canvas.save(slide_path)
        slide_files.append(slide_path)
        slide_index += 1

        children = focus_node.get("children", [])
        if children:
            print(f"Generating group slide {slide_index} for children of {focus_node['name']}")
            group_canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
            draw_ancestors(group_canvas, ancestors, HEADER_Y, LEVEL_HEIGHT, font, cache)
            parent_conn = draw_person(group_canvas, focus_node, HEADER_Y + len(ancestors) * LEVEL_HEIGHT, 
                                     True, font, cache, label=label)
            paste_y = HEADER_Y + (len(ancestors) + 1) * LEVEL_HEIGHT
            child_tiles = [get_tile(child, font, cache, label=child.get("customLabel", f"Child {i+1}")) 
                          for i, child in enumerate(children)]
            child_positions = paste_tiles(group_canvas, child_tiles, paste_y)

            draw_group = ImageDraw.Draw(group_canvas)
            horiz_y = paste_y - LINE_MARGIN
            draw_group.line([parent_conn, (parent_conn[0], horiz_y)], fill=LINE_COLOR, width=LINE_WIDTH)
            child_mid_points = [(x + w // 2, y) for (x, y, w, h) in child_positions]
            for cm in child_mid_points:
                draw_group.line([cm, (cm[0], horiz_y)], fill=LINE_COLOR, width=LINE_WIDTH)
            left_line = min(cm[0] for cm in child_mid_points)
            right_line = max(cm[0] for cm in child_mid_points)
            draw_group.line([(left_line, horiz_y), (right_line, horiz_y)], fill=LINE_COLOR, width=LINE_WIDTH)

            slide_path = os.path.join(slides_folder, f"slide_focused_{slide_index}.jpg")
            group_canvas.save(slide_path)
            slide_files.append(slide_path)
            slide_index += 1

            new_ancestors = ancestors + [(focus_node, label)]
            for i, child in enumerate(children):
                child_label = child.get("customLabel", f"Child {i+1}")
                if has_family_photo(child, cache):
                    print(f"Generating family slide {slide_index} for {child['name']}")
                    family_canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
                    draw_ancestors(family_canvas, new_ancestors, HEADER_Y, LEVEL_HEIGHT, font, cache)
                    family_photo_path = get_family_image(child, cache)
                    group_img = load_image(family_photo_path, default_size=(GROUP_PIC_MAX_WIDTH, GROUP_PIC_MAX_HEIGHT))
                    
                    if group_img:
                        group_img = resize_image(group_img, GROUP_PIC_MAX_WIDTH, GROUP_PIC_MAX_HEIGHT)
                        x = (CANVAS_WIDTH - group_img.width) // 2
                        y_group = HEADER_Y + len(new_ancestors) * LEVEL_HEIGHT
                        family_canvas.paste(group_img, (x, y_group))
                        draw_obj = ImageDraw.Draw(family_canvas)
                        y_text = y_group + group_img.height + LINE_SPACING
                        
                        line1 = f"{child['name']} ({child.get('customLabel', f'Child {i+1}')})"
                        if child.get("spouse", {}).get("name"):
                            line1 += f", {child['spouse']['name']} (Spouse)"
                        draw_centered_text(draw_obj, y_text, line1, label_font)
                        
                        y_text += LABEL_FONT_SIZE + LINE_SPACING
                        if child.get("children"):
                            child_labels = [f"{c['name']} (Child {j+1})" for j, c in enumerate(child["children"])]
                            line2 = ", ".join(child_labels)
                            draw_centered_text(draw_obj, y_text, line2, label_font)
                        
                        slide_path = os.path.join(slides_folder, f"slide_focused_{slide_index}.jpg")
                        family_canvas.save(slide_path)
                        slide_files.append(slide_path)
                        slide_index += 1
                else:
                    slide_index = recursive_focus(new_ancestors, child, slide_index, label=child_label)
        return slide_index

    for i, child in enumerate(root.get("children", [])):
        if child.get("spouse") or child.get("children"):
            initial_label = child.get("customLabel", f"Child {i+1}")
            slide_index = recursive_focus([(root, None)], child, slide_index, label=initial_label)
    return slide_files


def compute_subtree(node: dict, font: ImageFont.FreeTypeFont, cache: ImageCache) -> dict:
    """Recursively compute subtree dimensions and pre-render tiles."""
    tile = get_member_tile(node, font, cache)
    tile_width, tile_height = tile.size
    children_subtrees = []
    total_width = tile_width
    total_height = tile_height

    if node.get("children"):
        children_subtrees = [compute_subtree(child, font, cache) for child in node["children"]]
        children_width = sum(child["width"] for child in children_subtrees) + H_SPACER * (len(children_subtrees) - 1)
        children_height = max(child["height"] for child in children_subtrees) if children_subtrees else 0
        total_width = max(tile_width, children_width)
        total_height = tile_height + V_SPACER + children_height

    family_img = None
    if has_family_photo(node, cache):
        family_photo_path = get_family_image(node, cache)
        family_img = load_image(family_photo_path, default_size=(GROUP_PIC_MAX_WIDTH, GROUP_PIC_MAX_HEIGHT))
        if family_img:
            family_img = resize_image(family_img, GROUP_PIC_MAX_WIDTH, GROUP_PIC_MAX_HEIGHT)
            total_height += V_SPACER + family_img.height
            total_width = max(total_width, family_img.width)

    return {
        "tile": tile,
        "width": total_width,
        "height": total_height,
        "children": children_subtrees,
        "family_img": family_img,
        "original_width": tile_width,
    }


def draw_subtree(canvas: Image.Image, subtree: dict, x: int, y: int) -> tuple:
    """Recursively draw the subtree on the canvas."""
    tile = subtree["tile"]
    tile_width = subtree["original_width"]
    tile_height = tile.size[1]
    node_x = x + (subtree["width"] - tile_width) // 2
    canvas.paste(tile, (node_x, y))

    is_combined_tile = tile_width > 300
    if is_combined_tile:
        member_width = tile_width // 2 - SPACER // 2
        member_center = (node_x + member_width // 2, y + tile_height - LINE_MARGIN)
        spouse_center = (node_x + member_width + SPACER + member_width // 2, y + tile_height - LINE_MARGIN)
        draw = ImageDraw.Draw(canvas)
        spouse_line_y = y + tile_height // 2
        draw.line([(node_x + member_width + LINE_MARGIN, spouse_line_y),
                  (node_x + member_width + SPACER - LINE_MARGIN, spouse_line_y)], fill=LINE_COLOR, width=LINE_WIDTH)
        node_center = member_center
    else:
        node_center = (node_x + tile_width // 2, y + tile_height - LINE_MARGIN)

    children = subtree["children"]
    if children:
        children_y = y + tile_height + V_SPACER
        total_children_width = sum(child["width"] for child in children)
        total_spacing = H_SPACER * (len(children) - 1)
        children_start_x = x + (subtree["width"] - (total_children_width + total_spacing)) // 2
        child_centers = []
        current_x = children_start_x

        for child in children:
            child_center = draw_subtree(canvas, child, current_x, children_y)
            child_centers.append(child_center)
            current_x += child["width"] + H_SPACER

        draw = ImageDraw.Draw(canvas)
        horiz_y = children_y - V_SPACER // 2

        if len(children) == 1:
            child_x = child_centers[0][0]
            child_top_y = children_y + LINE_MARGIN
            draw.line([node_center, (node_center[0], horiz_y)], fill=LINE_COLOR, width=LINE_WIDTH)
            draw.line([(node_center[0], horiz_y), (child_x, horiz_y)], fill=LINE_COLOR, width=LINE_WIDTH)
            draw.line([(child_x, horiz_y), (child_x, child_top_y)], fill=LINE_COLOR, width=LINE_WIDTH)
        else:
            draw.line([node_center, (node_center[0], horiz_y)], fill=LINE_COLOR, width=LINE_WIDTH)
            left_x = child_centers[0][0]
            right_x = child_centers[-1][0]
            draw.line([(left_x, horiz_y), (right_x, horiz_y)], fill=LINE_COLOR, width=LINE_WIDTH)
            for child_center in child_centers:
                draw.line([(child_center[0], horiz_y), (child_center[0], children_y + LINE_MARGIN)], 
                          fill=LINE_COLOR, width=LINE_WIDTH)

    if subtree["family_img"]:
        family_img = subtree["family_img"]
        family_x = x + (subtree["width"] - family_img.width) // 2
        family_y = (children_y + max(child["height"] for child in children) + V_SPACER) if children else y + tile_height + V_SPACER
        canvas.paste(family_img, (family_x, family_y))

    return node_center


def create_complete_tree(config: dict, font: ImageFont.FreeTypeFont, cache: ImageCache) -> None:
    """Create a dynamically sized complete family tree image."""
    root = config["root"][0]
    subtree = compute_subtree(root, font, cache)
    padding = 50
    canvas = Image.new("RGB", (subtree["width"] + 2 * padding, subtree["height"] + 2 * padding), "white")
    draw_subtree(canvas, subtree, padding, padding)
    canvas.save(TREE_IMAGE_PATH)


def load_fonts() -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """Load and return required fonts."""
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        label_font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
    except IOError:
        print("Arial font not found, using default font")
        font = ImageFont.load_default()
        label_font = font
    return font, label_font


def create_final_slide(complete_tree: Image.Image, slides_folder: str) -> str:
    """Create the final slide with the complete family tree."""
    final_slide_path = os.path.join(slides_folder, "slide_final.jpg")
    final_canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    aspect_ratio = complete_tree.width / complete_tree.height

    if aspect_ratio > CANVAS_WIDTH / CANVAS_HEIGHT:
        new_width = CANVAS_WIDTH
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = CANVAS_HEIGHT
        new_width = int(new_height * aspect_ratio)

    resized_tree = complete_tree.resize((new_width, new_height), Image.Resampling.LANCZOS)
    x, y = (CANVAS_WIDTH - new_width) // 2, (CANVAS_HEIGHT - new_height) // 2
    final_canvas.paste(resized_tree, (x, y))
    final_canvas.save(final_slide_path)
    return final_slide_path


def main() -> None:
    """Main function to generate both the family video and complete tree image."""
    slides_folder = "slides"
    os.makedirs(slides_folder, exist_ok=True)

    try:
        with open(CONFIG_PATH) as f:
            config = preprocess_config(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading configuration: {e}")
        exit(1)

    cache = ImageCache()
    cache.initialize()
    font, label_font = load_fonts()

    # Generate tree and slides
    create_complete_tree(config, font, cache)
    slide_files = create_focused_slideshow(config, slides_folder, font, label_font, cache)

    # Add final slide
    with Image.open(TREE_IMAGE_PATH) as complete_tree:
        final_slide = create_final_slide(complete_tree, slides_folder)
        slide_files.append(final_slide)

    # Create video with audio
    try:
        clip = ImageSequenceClip(slide_files, fps=1/5)
        audio = AudioFileClip(MUSIC_PATH).with_duration(clip.duration)
        final_clip = clip.with_audio(audio)
        final_clip.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac")
    finally:
        shutil.rmtree(slides_folder)


if __name__ == "__main__":
    main()