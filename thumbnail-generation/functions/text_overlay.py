import sieve

@sieve.function(name="image_text_overlay",
                python_packages=["Pillow"],
                run_commands=["git clone https://github.com/ashvash182/workflow-custom-fonts"])
def text_overlay_func(base : sieve.Image, left : sieve.Image, right : sieve.Image, text : str, font_path : str) -> sieve.Image:
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

    # Define the text and font properties
    if len(text) > 40:
        font_size = 40
    elif len(text) > 20:
        font_size = 60
    else:
        font_size = 80
        
    font = ImageFont.truetype(font_path, font_size)  # Use an appropriate font file

    print('font successfully loaded!')

    # Calculate text size and position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (image_width - text_width) // 2  # Center the text horizontally
    y = (2 * image_height // 2.4) - (text_height // 2)
    
    # Determine background color at the text position
    bg_color = image.getpixel((x + text_width // 2, y + text_height // 2))
    bg_color_rgb = bg_color[:3]

    # Calculate the luminance of the background color
    luminance = colorsys.rgb_to_yiq(*bg_color_rgb)[0]

    # Determine the text color (black or white) based on background luminance
    if luminance > 128:
        text_color = "black"
    else:
        text_color = "white"

    # Add the text to the image
    draw.text((x, y), text, fill=text_color, font=font)
    
    # Create a border around the whole image
    bordered_image = ImageOps.expand(blurred_image)

    # Calculate the vertical position for the images above the text
    image_y = int(0.3*y)  # Place above the text and provide some space
    
    scaler = 2.3
    sep = 75

    # Load and resize the first image
    image1 = Image.open(left)
    image1 = image1.resize((int(image_width//scaler), int(image_height//scaler)))  # Resize as needed
    image1 = ImageOps.expand(image1, border=5, fill='black')

    # Load and resize the second image
    image2 = Image.open(right)
    image2 = image2.resize((int(image_width//scaler), int(image_height//scaler)))  # Resize as needed
    image2 = ImageOps.expand(image2, border=5, fill='black')
    
    # Create a new image to combine the base image, two side-by-side images, and text
    combined_image = Image.new('RGB', (image_width, image_height))

    # Paste the base image, two images, and text onto the new image
    combined_image.paste(bordered_image, (0, 0))
    combined_image.paste(image1, (int((image_width - sep - 2*(image_width//scaler))//2), image_y))
    combined_image.paste(image2, (int((image_width - sep - 2*(image_width//scaler))//2 + sep + image_width//scaler), image_y))
    
    combined_image.show()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_filename = temp_file.name
        combined_image.save(temp_filename)
        return sieve.Image(path=temp_filename)
