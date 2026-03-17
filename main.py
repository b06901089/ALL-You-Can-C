import sys
import cv2
import numpy as np
import functools

# cv2.imshow supports:                                                                                                                                                                                                                                                                                                                  
#   - 2D (H, W) — grayscale                                                                                                                                             
#   - 3D (H, W, 3) — BGR color                                                                                                                                          
#   - 3D (H, W, 4) — BGRA (with alpha channel)
def show_steps(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        prev_arrays = {}

        def tracer(frame, event, arg):
            if event == "line" and frame.f_code.co_name == func.__name__:
                for name, val in frame.f_locals.items():
                    if not isinstance(val, np.ndarray):
                        print(f"Error: must be an numpy.ndarray. target is tpye of {type(val)}")
                        continue
                    if not (val.ndim == 2 or (val.ndim == 3 and val.shape[2] in (1, 3, 4))):
                        print(f"Error: target's dimension \"{val.ndim}\" not supported in imshow()")
                        continue
                    
                    prev = prev_arrays.get(name)
                    if prev is None or not np.array_equal(prev, val):
                        prev_arrays[name] = val.copy()
                        dim_info = " x ".join(str(s) for s in val.shape)
                        cv2.imshow(f"[{name}]: ndim={val.ndim}, shape=({dim_info})", val)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

            return tracer

        sys.settrace(tracer)
        try:
            return func(*args, **kwargs)
        finally:
            sys.settrace(None)

    return wrapper


@show_steps
def process_image(input_path: str, output_path: str):
    image = cv2.imread(input_path)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(output_path, image)


if __name__ == "__main__":
    process_image("img/unsplash-cat-exmaple.jpg", "img/output.jpg")
