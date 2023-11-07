import sieve

@sieve.function(name="side_text_overlay",
                python_packages=["Pillow", "Wand"],
                system_packages=["imagemagick-6.q16"],
                run_commands=["mkdir -p /src/fonts", "git clone https://github.com/ashvash182/workflow-custom-fonts /src/fonts"])
def side_text_overlay(base : sieve.Image, left : sieve.Image, text : str, font_path : str) -> sieve.Image:
    from wand.image import Image
    from wand.display import display
    from wand.drawing import Drawing
    from wand.font import Font
    from wand.color import Color
    
    base_image_path = base.path
    left_image_path = left.path
    output_image_path = 'output.jpg'

    # Load the base image
    with Image(filename=base_image_path) as base_image:
        # Blur the base image
        base_image.blur(radius=15, sigma=110)

        # Load the left image
        with Image(filename=left_image_path) as left_image:
            # Resize the left image if needed
            left_image.resize(int(base_image.width / 2), int(base_image.height / 2))
            
            left_image.border('black', 5, 5)

            # Composite (overlay) the left image on the left half of the base image
            base_image.composite(left_image, left=int(base_image.width//3.5 - left_image.width//2), top=int(base_image.height//2 - left_image.height//2))
            
            width, height = left_image.height, left_image.width
            
            left, top = base_image.width//2 + width//2, base_image.height//2 - height//2
                
            with Drawing() as context:
                context.fill_color = 'None'
                context.rectangle(left=left, top=top, width=width, height=height)
                context(base_image)
            base_image.caption(text, left=left, top=top, width=width, height=height, font=Font(font_path, color=Color('white')), gravity='center')

            # Save the final image
            base_image.save(filename=output_image_path)
            
    return sieve.Image(path=output_image_path)

    # # Display the final image
    # display(Image(filename=output_image_path))