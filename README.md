# Family Tree Visualizer

A Python application that creates video slideshows from family tree data, visualizing family relationships with images and generating a complete family tree.

## Features

- Creates a video slideshow that walks through each branch of the family tree
- Generates a complete family tree visualization as a single image
- Supports custom profile pictures for individuals
- Includes family group photos in the slideshow
- Adds background music to the video slideshow
- Handles spouse relationships and multi-generational family trees
- Automatically assigns default images based on gender when custom images aren't available

## Requirements

- Python 3.7+
- Pillow >= 10.0.0 (for image processing)
- MoviePy >= 1.0.3 (for video creation)

## Installation

```bash
pip install -r requirements.txt
```

## File Structure

```
├── assets/
│   ├── audio/            # Background music files
│   │   └── background.mp3
│   └── images/
│       ├── defaults/     # Default images for males, females, and families
│       ├── families/     # Family group photos
│       └── profiles/     # Individual profile pictures
├── data/
│   └── family.json      # Family tree data structure
├── src/
│   └── family_tree.py   # Main application code
└── requirements.txt     # Python dependencies
```

## Usage

1. Prepare your family tree data in the `data/family.json` file following the structure shown in the example
2. Add profile pictures to `assets/images/profiles/` (named to match the person's name)
3. Add family photos to `assets/images/families/` (named to match the person's name)
4. Add a background music file as `assets/images/audio/background.mp3`
5. Run the application:

```bash
python src/family_tree.py
```

## Output

The application generates:

- `family_video.mp4`: A video slideshow that walks through the family tree
- `complete_family_tree.jpg`: A complete visualization of the entire family tree

## Data Format

The `family.json` file should contain a hierarchical structure of family members with the following format:

```json
{
  "root": [
    {
      "name": "Person Name",
      "gender": "male|female",
      "spouse": {
        "name": "Spouse Name",
        "gender": "male|female"
      },
      "children": [
        {
          "name": "Child Name",
          "gender": "male|female",
          "customLabel": "Optional Custom Label",
          "children": []
        }
      ]
    }
  ]
}
```

## Image Naming

Profile and family images should be named to match the person's name and placed in the appropriate directories:

- Individual profiles: `assets/images/profiles/personname.jpg`
- Family photos: `assets/images/families/personname.jpg`

Default images should be placed in `assets/images/defaults/` as:
- `male.jpg`: Default image for males
- `female.jpg`: Default image for females
- `family.jpg`: Default image for family groups