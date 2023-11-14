import sieve

@sieve.function(name="optimal_text_overlay",
                python_packages=["Pillow","Wand==0.6.11", "ghostscript"],
                system_packages=["imagemagick-6.q16"],
                run_commands=["mkdir -p /src/fonts", "git clone https://github.com/ashvash182/workflow-custom-fonts /src/fonts",
                              "mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.off"])
def optimal_text_placement(image : sieve.Image, text : str, font_path : str):
    from wand.image import Image
    from wand.display import display
    from wand.drawing import Drawing
    from wand.font import Font
    from wand.color import Color

    import wand

    object_detect = sieve.function.get('sieve/yolov8l')

    img_path = image.path
    bboxes = list(object_detect.run(sieve.File(path=img_path)))
    
    if len(bboxes) > 0:
        box_list = bboxes[0]['boxes']
    
    height_scaler = 2.2
    width_scaler = 2.5

    with Image(filename=img_path, resolution=50) as base:
        bcs = [
            [base.width//8, base.height//6, base.width//8 + base.width//width_scaler, base.height//6 + base.height//height_scaler],
            [3.5 * base.width//8, base.height//6, 3.5 * base.width//8 + base.width//width_scaler, base.height//6 + base.height//height_scaler],
            [3.5 * base.width//8, base.height//2, 3.5 * base.width//8 + base.width//width_scaler, base.height//2 + base.height//height_scaler],
            [base.width//8, base.height//2, base.width//8 + base.width//width_scaler, base.height//2 + base.height//height_scaler],
            [base.width//2 - base.width//4, base.height//2 - base.height//4, base.width//2 + base.width//4, base.height//2 + base.height//4],
            
            [base.width//10, base.height//6, base.width//10 + base.width//width_scaler, base.height//6 + base.height//height_scaler],
            [base.width//12, base.height//6, base.width//12 + base.width//width_scaler, base.height//6 + base.height//height_scaler],
            
            [4*base.width//10, base.height//6, 4*base.width//10 + base.width//width_scaler, base.height//6 + base.height//height_scaler],
            [4*base.width//12, base.height//6, 4*base.width//12 + base.width//width_scaler, base.height//6 + base.height//height_scaler],
            
            # wide bottoms
            [base.width//2 - base.width//3, base.height//2, base.width//2 + base.width//3, base.height//2 + base.height//(height_scaler*2)],
            [base.width//2 - base.width//3, base.height//1.5, base.width//2 + base.width//3, base.height//1.5 + base.height//(height_scaler*2)],
            
            # wider bottoms
            [base.width//2 - base.width//2.5, base.height//2, base.width//2 + base.width//2.5, base.height//2 + base.height//(height_scaler*2)],
            [base.width//2 - base.width//2.5, base.height//1.5, base.width//2 + base.width//2.5, base.height//1.5 + base.height//(height_scaler*2)],
            
            [base.width//2 - base.width//3, base.height//0.75, base.width//2 + base.width//3, base.height//0.75 + base.height//(height_scaler*2)],
            [base.width//2 - base.width//3, base.height//0.25, base.width//2 + base.width//3, base.height//0.25 + base.height//(height_scaler*2)]
        ]

        min_bbox_overlap = float('inf')
        best_cand = None
        
        if not bboxes:
            return None
        if bboxes:
            for i, cand in enumerate(bcs):
                bbox_overlap_area = 0

                for bbox in box_list:
                    bbox_area = (bbox['x2']-bbox['x1']) * (bbox['y2']-bbox['y1'])

                    w = None
                    h = None

                    if cand[0] < bbox['x1'] < cand[2]:
                        w = cand[2] - bbox['x1']
                    elif cand[0] < bbox['x2'] < cand[2]:
                        w = bbox['x2'] - cand[0]
                    elif bbox['x1'] < cand[0] < cand[2] < bbox['x2']:
                        w = cand[2] - cand[0]
                    else:
                        print('no width overlap')
                        continue

                    if cand[1] < bbox['y1'] < cand[3]:
                        h = cand[3] - bbox['y1']
                    elif cand[1] < bbox['y2'] < cand[3]:
                        h = bbox['y2'] - cand[1]
                    elif bbox['y1'] < cand[1] < cand[3] < bbox['y2']:
                        h = cand[3] - cand[1]
                    else:
                        print('no height overlap')
                        continue

                    bbox_overlap_area += (w*h)/bbox_area

                if bbox_overlap_area == 0:
                    print('zero overlap box found')
                    best_cand = i
                    break
                if bbox_overlap_area < min_bbox_overlap:
                    min_bbox_overlap = bbox_overlap_area
                    best_cand = i

        with Drawing() as context:
            context.fill_color = 'None'
            context(base)
        base.caption(text, left=int(bcs[best_cand][0]), top=int(bcs[best_cand][1]), width=int(bcs[best_cand][2]-bcs[best_cand][0]), height=int(bcs[best_cand][3]-bcs[best_cand][1]), font=Font(font_path, color=Color('white')), gravity='center')
        base.save(filename='./testing_optimality.png')
        
        yield sieve.Image(path='./testing_optimality.png')

    # display(Image(filename='./testing_optimality.png'))