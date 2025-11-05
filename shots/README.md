# Shots Directory

This directory contains shot folders for batch testing.

## Structure

Each shot folder must contain exactly these 3 files:

```
shots/
├── shot1_name/
│   ├── prompt.txt       # Positive prompt (text description of the shot)
│   ├── first_frame.png  # First frame of the video
│   └── last_frame.png   # Last frame of the video
├── shot2_name/
│   ├── prompt.txt
│   ├── first_frame.png
│   └── last_frame.png
└── ...
```

## File Requirements

### `prompt.txt`
- Plain text file
- Contains the positive prompt for this shot
- Can be multiple lines
- Example:
  ```
  A cinematic shot of a sunset over the ocean,
  with gentle waves and golden light reflecting on the water.
  Camera slowly pans right.
  ```

### `first_frame.png`
- PNG image file
- The starting frame of your video
- Can be any resolution (workflow will resize if needed)

### `last_frame.png`
- PNG image file
- The ending frame of your video
- Should match dimensions of first_frame.png

## Usage

1. **Create a shot folder:**
   ```bash
   mkdir shots/my_shot
   ```

2. **Add required files:**
   ```bash
   echo "Your prompt here" > shots/my_shot/prompt.txt
   cp /path/to/start.png shots/my_shot/first_frame.png
   cp /path/to/end.png shots/my_shot/last_frame.png
   ```

3. **Run shot batch:**
   ```bash
   python batch_runner_shots.py shot_batch_config.yaml
   ```

## Example Shot

See `example_shot/` for a template structure (you need to add actual images).

## Notes

- Shot folder names will be used in output organization
- Use descriptive names: `shot1_sunset`, `shot2_car_chase`, etc.
- All shots in this directory will be processed by the batch runner
- Missing any of the 3 required files = shot will be skipped with a warning
