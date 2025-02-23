import json
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from moviepy import ImageSequenceClip, AudioFileClip

# Configuration
CONFIG_PATH = "data/family.json"
MUSIC_PATH = "assets/audio/background.mp3"
OUTPUT_PATH = "family_video.mp4"
TREE_IMAGE_PATH = "complete_family_tree.jpg"
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 1600
HEADER_Y = 50
LEVEL_HEIGHT = 270
SPACER = 20
GROUP_PIC_MAX_WIDTH = 1000  # Maximum width for family photos
GROUP_PIC_MAX_HEIGHT = 600  # Maximum height for family photos
LABEL_FONT_SIZE = 24
LINE_SPACING = 40
LINE_MARGIN = 5  # margin to ensure lines stay outside the images
H_SPACER = 30  # horizontal spacing between sibling subtrees
V_SPACER = 50  # vertical spacing between levels

# Image paths
PROFILE_IMAGES_PATH = "assets/images/profiles"
FAMILY_IMAGES_PATH = "assets/images/families"
DEFAULT_IMAGES_PATH = "assets/images/defaults"


def preprocess_config(config: dict) -> dict:
    """
    Preprocess the configuration to ensure spouse gender is set appropriately.
    """

    def process_node(node: dict) -> None:
        if "spouse" in node and node["spouse"]:
            spouse = node["spouse"]
            if not spouse.get("gender"):
                spouse["gender"] = (
                    "female" if node.get("gender", "").lower() == "male" else "male"
                )
        for child in node.get("children", []):
            process_node(child)

    for root in config.get("root", []):
        process_node(root)
    return config


def get_image_path(name: str, image_type: str = "profile") -> str:
    """
    Retrieve the cached image path for a given name and image type.
    """
    key = name.strip().lower()
    if image_type == "profile":
        return profile_files.get(key)
    return family_files.get(key)


def get_default_image(gender: str) -> str:
    """
    Retrieve the default image path for a given gender.
    """
    gender = gender.lower()
    return (
        default_male
        if gender == "male"
        else default_female if gender == "female" else None
    )


def get_family_image(member: dict) -> str:
    """
    Retrieve the family photo for a member, if available.
    """
    if member.get("hasFamily") is False:
        return None
    family_photo = get_image_path(member["name"], "family")
    if family_photo:
        return family_photo
    return default_family if member.get("hasFamily") and default_family else None


def has_family_photo(member: dict) -> bool:
    """
    Check if the member has a family photo.
    """
    return get_family_image(member) is not None


def resize_image(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """
    Resize image preserving aspect ratio to fit within max dimensions.
    """
    aspect_ratio = img.width / img.height
    if aspect_ratio > max_width / max_height:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def get_tile(
    member: dict, config: dict, font: ImageFont.FreeTypeFont, label: str = None
) -> Image.Image:
    """
    Create a tile for a member with their profile image and label.
    Dynamically adjusts the tile width so the full label is always visible.
    """
    base_img_size = 200
    text_padding = 10
    base_name = member["name"]
    display_name = f"{base_name} ({label})" if label else base_name

    # Measure text width.
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    text_bbox = temp_draw.textbbox((0, 0), display_name, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    tile_width = max(base_img_size, text_width + 2 * text_padding)
    tile_height = base_img_size + 50  # extra space for label below image
    tile = Image.new("RGB", (tile_width, tile_height), "white")

    # Attempt to open the profile image.
    img = None
    profile_path = get_image_path(base_name, "profile")
    if profile_path:
        try:
            img = Image.open(profile_path)
        except Exception as e:
            print(f"Error opening image {profile_path}: {e}")

    if not img:
        default_path = get_default_image(member.get("gender", "male"))
        if default_path:
            try:
                img = Image.open(default_path)
            except Exception as e:
                print(f"Error opening default image {default_path}: {e}")

    if img:
        img = img.resize((base_img_size, base_img_size))
    else:
        # Create a placeholder image.
        color = (
            "dodgerblue"
            if member.get("gender", "male").lower() == "male"
            else "hotpink"
        )
        img = Image.new("RGB", (base_img_size, base_img_size), color)
        d = ImageDraw.Draw(img)
        d.text(
            (base_img_size // 2, base_img_size // 2),
            base_name,
            fill="white",
            anchor="mm",
            font=font,
        )

    # Paste the image centered horizontally.
    img_x = (tile_width - base_img_size) // 2
    tile.paste(img, (img_x, 0))
    d = ImageDraw.Draw(tile)
    text_y = base_img_size + ((tile_height - base_img_size) // 2)
    d.text(
        (tile_width // 2, text_y), display_name, fill="black", font=font, anchor="mm"
    )
    return tile


def get_member_tile(
    member: dict, config: dict, font: ImageFont.FreeTypeFont, label: str = None
) -> Image.Image:
    """
    Create a combined tile for a member.
    If the member has a spouse, combine their tiles side by side.
    """
    if member.get("spouse"):
        tile1 = get_tile(member, config, font, label=label)
        tile2 = get_tile(member["spouse"], config, font, label="Spouse")
        height = max(tile1.height, tile2.height)
        width = tile1.width + SPACER + tile2.width
        combined = Image.new("RGB", (width, height), "white")
        y1 = (height - tile1.height) // 2
        y2 = (height - tile2.height) // 2
        combined.paste(tile1, (0, y1))
        combined.paste(tile2, (tile1.width + SPACER, y2))
        return combined
    return get_tile(member, config, font, label=label)


def paste_tiles(canvas: Image.Image, tiles: list, y: int) -> list:
    """
    Paste tiles horizontally centered on the canvas.
    Returns a list of (x, y, width, height) for each pasted tile.
    """
    positions = []
    total_width = sum(tile.width for tile in tiles) + SPACER * (len(tiles) - 1)
    start_x = (CANVAS_WIDTH - total_width) // 2
    for tile in tiles:
        canvas.paste(tile, (start_x, y))
        positions.append((start_x, y, tile.width, tile.height))
        start_x += tile.width + SPACER
    return positions


def draw_person(
    canvas: Image.Image,
    node: dict,
    y: int,
    show_spouse: bool,
    config: dict,
    font: ImageFont.FreeTypeFont,
    is_root: bool = False,
    label: str = None,
) -> tuple:
    """
    Draw a person (and their spouse, if applicable) on the canvas.
    Returns the connection point (x, y) for drawing further vertical lines.
    """
    if show_spouse and node.get("spouse"):
        main_label = label if label is not None else (None if is_root else "Child")
        tiles = [
            get_tile(node, config, font, label=main_label),
            get_tile(node["spouse"], config, font, label="Spouse"),
        ]
    else:
        tiles = [get_tile(node, config, font, label=label)]
    positions = paste_tiles(canvas, tiles, y)
    left = positions[0][0]
    right = positions[-1][0] + positions[-1][2]
    mid_x = (left + right) // 2
    bottom_y = max(pos[1] + pos[3] for pos in positions)
    return (mid_x, bottom_y + LINE_MARGIN)


def draw_ancestors(
    canvas: Image.Image,
    ancestors: list,
    header_y: int,
    level_height: int,
    config: dict,
    font: ImageFont.FreeTypeFont,
) -> None:
    """
    Draw ancestors on the canvas.
    """
    for i, (node, lbl) in enumerate(ancestors):
        is_root = i == 0
        draw_person(
            canvas,
            node,
            header_y + i * level_height,
            True,
            config,
            font,
            is_root,
            label=lbl,
        )


def draw_centered_text(
    draw: ImageDraw.ImageDraw, y: int, text: str, font: ImageFont.FreeTypeFont
) -> None:
    """
    Draw centered text on the canvas.
    """
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text(((CANVAS_WIDTH - text_width) // 2, y), text, fill="black", font=font)


def create_focused_slideshow(
    config: dict,
    slides_folder: str,
    font: ImageFont.FreeTypeFont,
    label_font: ImageFont.FreeTypeFont,
) -> list:
    """
    Create a slideshow focusing on family members and their relationships.
    """
    os.makedirs(slides_folder, exist_ok=True)
    slide_files = []
    root = config["root"][0]

    print("Generating slide 0: Root only")
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw_person(canvas, root, HEADER_Y, True, config, font, is_root=True)
    slide_path = os.path.join(slides_folder, "slide_focused_0.jpg")
    canvas.save(slide_path)
    slide_files.append(slide_path)

    print("Generating slide 1: Root with children")
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw_person(canvas, root, HEADER_Y, True, config, font, is_root=True)
    children = root.get("children", [])
    if children:
        child_tiles = [
            get_tile(child, config, font, label=f"Child {i+1}")
            for i, child in enumerate(children)
        ]
        paste_tiles(canvas, child_tiles, HEADER_Y + LEVEL_HEIGHT)
    slide_path = os.path.join(slides_folder, "slide_focused_1.jpg")
    canvas.save(slide_path)
    slide_files.append(slide_path)

    slide_index = 2

    def recursive_focus(
        ancestors: list, focus_node: dict, slide_index: int, label: str = None
    ) -> int:
        nonlocal slide_files
        print(f"Generating base slide {slide_index} for {focus_node['name']}")
        base_canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
        draw_ancestors(base_canvas, ancestors, HEADER_Y, LEVEL_HEIGHT, config, font)
        parent_conn = draw_person(
            base_canvas,
            focus_node,
            HEADER_Y + len(ancestors) * LEVEL_HEIGHT,
            True,
            config,
            font,
            label=label,
        )
        slide_path = os.path.join(slides_folder, f"slide_focused_{slide_index}.jpg")
        base_canvas.save(slide_path)
        slide_files.append(slide_path)
        slide_index += 1

        children = focus_node.get("children", [])
        if children:
            print(
                f"Generating group slide {slide_index} for children of {focus_node['name']}"
            )
            group_canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
            draw_ancestors(
                group_canvas, ancestors, HEADER_Y, LEVEL_HEIGHT, config, font
            )
            parent_conn = draw_person(
                group_canvas,
                focus_node,
                HEADER_Y + len(ancestors) * LEVEL_HEIGHT,
                True,
                config,
                font,
                label=label,
            )
            paste_y = HEADER_Y + (len(ancestors) + 1) * LEVEL_HEIGHT
            child_tiles = [
                get_tile(child, config, font, label=f"Child {i+1}")
                for i, child in enumerate(children)
            ]
            child_positions = paste_tiles(group_canvas, child_tiles, paste_y)

            draw_group = ImageDraw.Draw(group_canvas)
            horiz_y = paste_y - LINE_MARGIN
            draw_group.line(
                [parent_conn, (parent_conn[0], horiz_y)], fill="gray", width=2
            )
            child_mid_points = [(x + w // 2, y) for (x, y, w, h) in child_positions]
            for cm in child_mid_points:
                draw_group.line([cm, (cm[0], horiz_y)], fill="gray", width=2)
            left_line = min(cm[0] for cm in child_mid_points)
            right_line = max(cm[0] for cm in child_mid_points)
            draw_group.line(
                [(left_line, horiz_y), (right_line, horiz_y)], fill="gray", width=2
            )

            slide_path = os.path.join(slides_folder, f"slide_focused_{slide_index}.jpg")
            group_canvas.save(slide_path)
            slide_files.append(slide_path)
            slide_index += 1

            new_ancestors = ancestors + [(focus_node, label)]
            for i, child in enumerate(children):
                child_label = f"Child {i+1}"
                if has_family_photo(child):
                    print(f"Generating family slide {slide_index} for {child['name']}")
                    family_canvas = Image.new(
                        "RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white"
                    )
                    draw_ancestors(
                        family_canvas,
                        new_ancestors,
                        HEADER_Y,
                        LEVEL_HEIGHT,
                        config,
                        font,
                    )
                    family_photo_path = get_family_image(child)
                    try:
                        group_img = Image.open(family_photo_path)
                        group_img = resize_image(
                            group_img, GROUP_PIC_MAX_WIDTH, GROUP_PIC_MAX_HEIGHT
                        )
                    except Exception as e:
                        print(f"Error opening family photo {family_photo_path}: {e}")
                        group_img = Image.new(
                            "RGB", (GROUP_PIC_MAX_WIDTH, GROUP_PIC_MAX_HEIGHT), "gray"
                        )
                    x = (CANVAS_WIDTH - group_img.width) // 2
                    y_group = HEADER_Y + len(new_ancestors) * LEVEL_HEIGHT
                    family_canvas.paste(group_img, (x, y_group))
                    draw_obj = ImageDraw.Draw(family_canvas)
                    y_text = y_group + group_img.height + LINE_SPACING
                    line1 = f"{child['name']} (Child {i+1})"
                    if child.get("spouse", {}).get("name"):
                        line1 += f", {child['spouse']['name']} (Spouse)"
                    draw_centered_text(draw_obj, y_text, line1, label_font)
                    y_text += LABEL_FONT_SIZE + LINE_SPACING
                    if child.get("children"):
                        child_labels = [
                            f"{c['name']} (Child {j+1})"
                            for j, c in enumerate(child["children"])
                        ]
                        line2 = ", ".join(child_labels)
                        draw_centered_text(draw_obj, y_text, line2, label_font)
                        y_text += LABEL_FONT_SIZE + LINE_SPACING
                    slide_path = os.path.join(
                        slides_folder, f"slide_focused_{slide_index}.jpg"
                    )
                    family_canvas.save(slide_path)
                    slide_files.append(slide_path)
                    slide_index += 1
                else:
                    slide_index = recursive_focus(
                        new_ancestors, child, slide_index, label=child_label
                    )
        return slide_index

    for i, child in enumerate(root.get("children", [])):
        if child.get("spouse") or child.get("children"):
            slide_index = recursive_focus(
                [(root, None)], child, slide_index, label=f"Child {i+1}"
            )
    return slide_files


def compute_subtree(node: dict, config: dict, font: ImageFont.FreeTypeFont) -> dict:
    """
    Recursively compute the subtree dimensions and pre-render the node tile.
    Returns a dict with keys: tile, width, height, children, and family_img.
    """
    tile = get_member_tile(node, config, font)
    tile_width, tile_height = tile.size
    children_subtrees = []
    subtree_width = tile_width
    subtree_height = tile_height

    if node.get("children"):
        for child in node["children"]:
            child_tree = compute_subtree(child, config, font)
            children_subtrees.append(child_tree)
        children_width = sum(
            child["width"] for child in children_subtrees
        ) + H_SPACER * (len(children_subtrees) - 1)
        children_height = max(child["height"] for child in children_subtrees)
        subtree_width = max(tile_width, children_width)
        subtree_height = tile_height + V_SPACER + children_height

    family_img = None
    if has_family_photo(node):
        family_photo_path = get_family_image(node)
        try:
            family_img = Image.open(family_photo_path)
            family_img = resize_image(
                family_img, GROUP_PIC_MAX_WIDTH, GROUP_PIC_MAX_HEIGHT
            )
        except Exception as e:
            print(f"Error opening family photo {family_photo_path}: {e}")
            family_img = None
        if family_img:
            subtree_height += V_SPACER + family_img.height
            subtree_width = max(subtree_width, family_img.width)

    return {
        "tile": tile,
        "width": subtree_width,
        "height": subtree_height,
        "children": children_subtrees,
        "family_img": family_img,
    }


def draw_subtree(
    canvas: Image.Image, subtree: dict, top_left_x: int, top_left_y: int
) -> None:
    """
    Recursively draw the subtree on the canvas starting at (top_left_x, top_left_y).
    """
    tile = subtree["tile"]
    tile_width, tile_height = tile.size
    allocated_width = subtree["width"]
    node_x = top_left_x + (allocated_width - tile_width) // 2
    node_y = top_left_y
    canvas.paste(tile, (node_x, node_y))
    current_bottom = node_y + tile_height

    if subtree["children"]:
        total_children_width = sum(
            child["width"] for child in subtree["children"]
        ) + H_SPACER * (len(subtree["children"]) - 1)
        children_top = current_bottom + V_SPACER
        start_x = top_left_x + (allocated_width - total_children_width) // 2
        child_centers = []
        for child in subtree["children"]:
            draw_subtree(canvas, child, start_x, children_top)
            child_center = (
                start_x + child["width"] // 2,
                children_top + child["height"] // 2,
            )
            child_centers.append(child_center)
            start_x += child["width"] + H_SPACER

        d = ImageDraw.Draw(canvas)
        parent_center = (top_left_x + allocated_width // 2, node_y + tile_height)
        children_line_y = children_top
        d.line(
            [parent_center, (parent_center[0], children_line_y)], fill="gray", width=2
        )
        if len(child_centers) > 1:
            left_x = min(x for x, y in child_centers)
            right_x = max(x for x, y in child_centers)
            d.line(
                [(left_x, children_line_y), (right_x, children_line_y)],
                fill="gray",
                width=2,
            )
            for cx, _ in child_centers:
                d.line(
                    [(cx, children_line_y), (cx, children_top)], fill="gray", width=2
                )
        current_bottom = children_top + max(
            child["height"] for child in subtree["children"]
        )

    if subtree["family_img"]:
        family_img = subtree["family_img"]
        family_img_width, family_img_height = family_img.size
        family_top = current_bottom + V_SPACER
        family_x = top_left_x + (allocated_width - family_img_width) // 2
        canvas.paste(family_img, (family_x, family_top))


def create_complete_tree(
    config: dict, font: ImageFont.FreeTypeFont, label_font: ImageFont.FreeTypeFont
) -> None:
    """
    Create a dynamically sized complete family tree image.
    """
    root = config["root"][0]
    subtree = compute_subtree(root, config, font)
    padding = 50
    canvas_width = subtree["width"] + 2 * padding
    canvas_height = subtree["height"] + 2 * padding
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw_subtree(canvas, subtree, padding, padding)
    canvas.save(TREE_IMAGE_PATH)


def main() -> None:
    """
    Main function to generate both the family video and complete tree image.
    """
    slides_folder = "slides"
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
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    profile_files = {
        os.path.splitext(f)[0].lower(): os.path.join(PROFILE_IMAGES_PATH, f)
        for f in os.listdir(PROFILE_IMAGES_PATH)
        if os.path.splitext(f)[1].lower() in allowed_extensions
    }
    family_files = {
        os.path.splitext(f)[0].lower(): os.path.join(FAMILY_IMAGES_PATH, f)
        for f in os.listdir(FAMILY_IMAGES_PATH)
        if os.path.splitext(f)[1].lower() in allowed_extensions
    }

    try:
        font = ImageFont.truetype("arial.ttf", 18)
        label_font = ImageFont.truetype("arial.ttf", LABEL_FONT_SIZE)
    except IOError:
        font = ImageFont.load_default()
        label_font = font

    global default_male, default_female, default_family
    default_male = os.path.join(DEFAULT_IMAGES_PATH, "male.jpg")
    default_female = os.path.join(DEFAULT_IMAGES_PATH, "female.jpg")
    default_family = os.path.join(DEFAULT_IMAGES_PATH, "family.jpg")
    if not os.path.exists(default_male):
        default_male = None
    if not os.path.exists(default_female):
        default_female = None
    if not os.path.exists(default_family):
        default_family = None

    # Generate the video slides
    slide_files = create_focused_slideshow(config, slides_folder, font, label_font)

    # Generate the complete tree image
    create_complete_tree(config, font, label_font)

    # Create the video
    clip = ImageSequenceClip(slide_files, fps=1 / 5)
    audio = AudioFileClip(MUSIC_PATH).with_duration(clip.duration)
    final_clip = clip.with_audio(audio)
    final_clip.write_videofile(OUTPUT_PATH, codec="libx264", audio_codec="aac")

    shutil.rmtree(slides_folder)


if __name__ == "__main__":
    main()
