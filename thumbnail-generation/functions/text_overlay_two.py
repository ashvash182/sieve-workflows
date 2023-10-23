import sieve

@sieve.function(name="image_text_overlay_two",
                python_packages=["Pillow"],
                run_commands=["git clone https://github.com/ashvash182/workflow-custom-fonts"])
def text_overlay_func_two(base : sieve.Image, left : sieve.Image, right : sieve.Image, text : str, font_path : str) -> sieve.Image:
    from PIL import Image, ImageFilter, ImageOps, ImageFont, ImageDraw
    import colorsys

    base = base.path
    left = left.path
    right = right.path

    import tempfile
    # Load the image
    if not base:
        return
    image = Image.open(base)

    # Apply a Gaussian blur to the image
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=10))

    # Get the dimensions of the image
    image_width, image_height = image.size

    # Create a drawing context
    draw = ImageDraw.Draw(blurred_image)