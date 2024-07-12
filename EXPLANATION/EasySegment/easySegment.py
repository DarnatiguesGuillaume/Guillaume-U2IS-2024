from PIL import Image, ImageTk
from tkinter import ttk
import tkinter as tk
import numpy as np
import torch
import cv2
import os


import onnxruntime
from onnxruntime.quantization import QuantType
from segment_anything.utils.onnx import SamOnnxModel
from segment_anything import sam_model_registry, SamPredictor
from onnxruntime.quantization.quantize import quantize_dynamic

"""    This program lets you create masks for images of the same format of resolution

    Choose the labels by putting them in a .txt, one per line, no space, then put its path in           Opt[0]
    Then choose the input directory, which is the folder in which all images will be found              Opt[1]
    Output directory can be chosen as well                                                              Opt[2]
    You also have to specify the location of the SAM checkpoint used and model type                     Opt[3], Opt[4]
    Additional changes are sam model type l46, masks color l287 and the output name l186, cuda or cpu   Opt[5]
    
    By default masks are saved with the image name, the mask name and an id in case you want to save a single mask in two times (if you forgot
     anything for exemple) but it is recommended to finish a single mask of an image in one session as that id is reset on restart of the program.
     masks are saved as [[array of true/false]] in numpy format as it is done in SAM, so that you can fuse two with some really simple "&" operations.
"""

Opt = [
    'label_list.txt', 'images/', 'masks/', "SAM_checkpoint/sam_vit_h_4b8939.pth", "vit_h", "cpu" 
]

label_list_path = Opt[0]
with open(label_list_path, 'r') as file:
    label_list = file.read().splitlines()

input_dir = Opt[1]
image_list = os.listdir(input_dir)

#SAM<
input_point = np.empty((0, 2))
input_label = np.empty((0,))

sam_checkpoint = Opt[3]
model_type = Opt[4]
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
onnx_model_path = None  # Set to use an already exported model, then skip to the next section.
#>SAM


class DrawingApp(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.master.title("Drawing App")

        self.grid_width = 12
        self.i = 0
        

        master.grid_rowconfigure(0, weight=0)
        master.grid_rowconfigure(1, weight=0)
        master.grid_rowconfigure(2, weight=1)

        for i in range(self.grid_width):
            master.grid_columnconfigure(i, weight=1)

        master.geometry("1600x900")
        
        # Variables
        self.mask_save_id = 0
        self.real_x = 0
        self.real_y = 0
        self.predictor = None
        self.image_ratio = (1,1) 
        self.mode = tk.IntVar(value=1)
        self.erase_mode = tk.BooleanVar(value=False)
        self.size = tk.IntVar(value=5)
        self.single_input_mode = tk.BooleanVar(value=False)
        self.save_choice = tk.StringVar(value="0")
        self.load_choice = tk.StringVar(value="0")
        self.fuse_choice = tk.StringVar(value="0")
        self.negative_fuse_mode = tk.BooleanVar(value=False)
        self.negative_input_mode = tk.BooleanVar(value=False)
        
        # UI Elements
        tk.Radiobutton(master, text="Segmentation", variable=self.mode, value = 1).grid(row=0, column=self.i, sticky="we")
        tk.Radiobutton(master, text="Brush", variable=self.mode, value = 0).grid(row=1, column=self.i, sticky="we")
        self.i +=1

        tk.Checkbutton(master, text="Single Input Mode", variable=self.single_input_mode).grid(row=0, column=self.i, sticky="we")
        tk.Checkbutton(master, text="Erase Mode", variable=self.erase_mode).grid(row=1, column=self.i, sticky="we")
        self.i +=1

        tk.Checkbutton(master, text="Negative Input Mode", variable=self.negative_input_mode).grid(row=0, column=self.i, columnspan=1)
        tk.Button(master, text="Save !", command=self.save_image).grid(row=1, column=self.i)
        self.i +=1

        tk.Label(master, text="Select Label:").grid(row=0, column=self.i, sticky="e")
        tk.Label(master, text="Brush Size:").grid(row=1, column=self.i, sticky="e")
        self.i +=1

        self.label_combobox = ttk.Combobox(master, values=label_list)
        self.label_combobox.grid(row=0, column=self.i, sticky="we")
        self.label_combobox.set(label_list[0])
        tk.Entry(master, textvariable=self.size, width=5).grid(row=1, column=self.i, sticky="w")
        self.label_combobox.config(width=10)
        self.size.set(15)
        self.i +=1

        tk.Label(master, text="Select Image:").grid(row=0, column=self.i, sticky="e")
        tk.Label(master, text="Load Choice:").grid(row=1, column=self.i, sticky="e")
        self.i +=1

        self.image_combobox = ttk.Combobox(master, values=image_list)
        self.image_combobox.grid(row=0, column=self.i, sticky="we")
        self.image_combobox.set(image_list[0])
        self.image_combobox.config(width=10)
        ###
        self.load_combobox = ttk.Combobox(master, textvariable=self.load_choice, values=[str(i) for i in range(0,10)] + ["last"])
        self.load_combobox.grid(row=1, column=self.i, sticky="w")
        self.load_combobox.config(width=3)
        self.i +=1
        
        tk.Button(master, text="Process Current Image", command=self.process_image).grid(row=0, column=self.i, columnspan=2)
        tk.Label(master, text="Save Choice:").grid(row=1, column=self.i, sticky="e")
        self.i +=1

        self.save_combobox = ttk.Combobox(master, textvariable=self.save_choice, values=[str(i) for i in range(0,10)])
        self.save_combobox.grid(row=1, column=self.i, sticky="w")
        self.save_combobox.config(width=3)
        self.i +=1

        tk.Checkbutton(master, text="Negative Fuse Mode", variable=self.negative_fuse_mode).grid(row=0, column=self.i, columnspan=2)
        tk.Label(master, text="Fuse Choice:").grid(row=1, column=self.i, sticky="e")
        self.i +=1

        self.fuse_combobox = ttk.Combobox(master, textvariable=self.fuse_choice, values=[str(i) for i in range(0,10)])
        self.fuse_combobox.grid(row=1, column=self.i, sticky="w")
        self.fuse_combobox.config(width=3)
        self.i +=1

        tk.Label(master, text=" ").grid(row=0, column=self.i, columnspan=2, rowspan=2)


        # image setup on startup
        self.window_on_startup()

        self.image_combobox.bind("<<ComboboxSelected>>",self.load_and_display_image)
        master.bind("<Configure>", self.on_window_resize)
        self.image_label.bind("<Button-1>", self.left_click)
        self.image_label.bind("<B1-Motion>", self.left_click)
        self.load_combobox.bind("<<ComboboxSelected>>", self.load_state)
        self.save_combobox.bind("<<ComboboxSelected>>", self.save_state)
        self.fuse_combobox.bind("<<ComboboxSelected>>", self.fuse_with_state)

    def fuse_with_state(self, event):
        self.savestates[9] = np.copy(self.masks)

        selected_value = self.fuse_combobox.get()
        state_number = int(selected_value)
        if self.negative_fuse_mode.get()==True:
            self.masks = self.masks & ~self.savestates[state_number-1]
        else:
            self.masks = self.masks | self.savestates[state_number-1]
        self.window_update()
    
    def load_state(self, event):
        selected_value = self.load_combobox.get()
        if selected_value == "last":
            self.masks = np.copy(self.savestates[9])
        else:
            state_number = int(selected_value)
            self.masks = np.copy(self.savestates[state_number-1])
        self.window_update()

    def save_state(self, event):
        selected_value = self.save_combobox.get()
        state_number = int(selected_value)
        self.savestates[state_number-1] = np.copy(self.masks)



    def save_image(self):
        out_name = Opt[2] + self.image_combobox.get().rsplit('.', 1)[0] + "_" + self.label_combobox.get() + "_" + f"{self.mask_save_id:08}" + ".npy"
        np.save(out_name, self.masks)
        print("Saved as: ",out_name)
        self.mask_save_id +=1
    
    def draw_circle(self):
        y, x = np.ogrid[:self.masks.shape[2], :self.masks.shape[3]]
        distances = (x - self.real_x)**2 + (y - self.real_y)**2

        if self.erase_mode.get()==False:
            circle = distances <= self.size.get()**2
            circle = [[circle]]
            self.masks = self.masks | circle
        else:
            circle = distances > self.size.get()**2
            circle = [[circle]]
            self.masks = self.masks & circle

    def left_click(self, event):
        global input_point
        global input_label
        self.savestates[9] = np.copy(self.masks)
        self.get_real_coords(event.x,event.y)
        if (self.mode.get()==1) and (self.predictor is not None):
            if self.negative_input_mode.get()==False:
                if self.single_input_mode.get()==True:
                    input_point = np.array([[self.real_x, self.real_y]])
                    input_label = np.array([1])
                else:
                    input_point = np.append(input_point, [[self.real_x, self.real_y]], axis=0)
                    input_label = np.append(input_label, 1)
            else:
                input_point = np.append(input_point, [[self.real_x, self.real_y]], axis=0)
                input_label = np.append(input_label, 0)
            self.segment_with_input()
        else:
            self.draw_circle()
        self.window_update()

#SAM<
    def segment_with_input(self):
        global input_point, input_label  

        onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

        onnx_coord = self.predictor.transform.apply_coords(onnx_coord, self.cv2_image.shape[:2]).astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": self.image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.cv2_image.shape[:2], dtype=np.float32)
        }

        self.masks, _, low_res_logits = self.ort_session.run(None, ort_inputs)
        self.masks = self.masks > self.predictor.model.mask_threshold

    def segment_image(self):
            #image,onnx_model_path,sam):
        global onnx_model_path
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.ort_session = onnxruntime.InferenceSession(onnx_model_path)

        print("Starting prediction...", end="\t")
        sam.to(device=Opt[5])
        #cpu
        self.predictor = SamPredictor(sam)
        self.predictor.set_image(self.cv2_image)
        self.image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        self.image_embedding.shape
        print("...prediction Ended !")
        #return [image_embedding,ort_session,predictor] 
#>SAM
        
        
    def window_on_startup(self):
        self.cv2_image = cv2.imread(os.path.join(input_dir, self.image_combobox.get()))
        self.cv2_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)
        self.image_ratio = self.cv2_image.shape
        self.masks = np.array([[np.zeros_like(self.cv2_image[..., 0], dtype=bool)]])
        self.savestates = np.array([np.copy(self.masks) for _ in range(10)])
        
        # Calculate the initial window size
        _, _, self.width, self.height = self.master.grid_bbox(0, 2, self.grid_width - 1, 2)

        # Ensure width and height are positive
        if self.width > 0 and self.height > 0:
            # Calculate the resized image dimensions based on the window size
            if int(self.width * self.image_ratio[0] / self.image_ratio[1]) > self.height:
                self.width = int(self.height * self.image_ratio[1] / self.image_ratio[0])
            else:
                self.height = int(self.width * self.image_ratio[0] / self.image_ratio[1])
        else:
            self.width = self.image_ratio[1]
            self.height = self.image_ratio[0]
        color = np.array([255, 144, 30])
        h, w = self.masks.shape[-2:]
        mask_image = self.masks.reshape(h, w, 1) * color.reshape(1, 1, -1).astype(np.uint8)
        result = cv2.addWeighted(self.cv2_image, 1, mask_image, 0.6, 0)

        resized_result = cv2.resize(result, (self.width, self.height))
        result_image = Image.fromarray(resized_result)
        result_tk_image = ImageTk.PhotoImage(result_image)

        # Update the label with the resized image
        self.image_label = tk.Label(self.master, image=result_tk_image)
        self.image_label.grid(row=2, column=0, columnspan=self.grid_width, padx=15, pady=15)
        self.image_label.image = result_tk_image





    def on_window_resize(self, event):
        # Schedule the action to be performed after 100 milliseconds (0.1 seconds)
        self.after(10, self.resize_image)

    def resize_image(self):
        # Get the current window size
        _, _, self.width, self.height = self.master.grid_bbox(0, 2, self.grid_width - 1, 2)

        if self.width > 0 and self.height > 0:
            # Calculate the resized image dimensions based on the window size
            if int(self.width * self.image_ratio[0] / self.image_ratio[1]) > self.height:
                self.width = int(self.height * self.image_ratio[1] / self.image_ratio[0])
            else:
                self.height = int(self.width * self.image_ratio[0] / self.image_ratio[1])
        else:
            self.width = self.image_ratio[1]
            self.height = self.image_ratio[0]

        self.window_update()

    def window_update(self):
        color = np.array([255, 144, 30])
        h, w = self.masks.shape[-2:]
        mask_image = self.masks.reshape(h, w, 1) * color.reshape(1, 1, -1).astype(np.uint8)
        result = cv2.addWeighted(self.cv2_image, 1, mask_image, 0.6, 0)

        resized_result = cv2.resize(result, (self.width, self.height))
        result_image = Image.fromarray(resized_result)
        result_tk_image = ImageTk.PhotoImage(result_image)
        self.image_label.configure(image=result_tk_image)
        self.image_label.image = result_tk_image

    def get_real_coords(self,x,y):
        self.real_x = int(x*self.cv2_image.shape[1] /self.width)
        self.real_y = int(y*self.cv2_image.shape[0] /self.height)

    def process_image(self):
        #self.mask_save_id = 0
        self.segment_image()
        #[self.image_embedding,self.ort_session,self.predictor] = segment_image(self.cv2_image,onnx_model_path,sam)

    def load_and_display_image(self, event):
        self.cv2_image = cv2.imread(os.path.join(input_dir, self.image_combobox.get()))
        self.cv2_image = cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2RGB)
        self.image_ratio = self.cv2_image.shape
        self.masks = np.array([[np.zeros_like(self.cv2_image[..., 0], dtype=bool)]])
        self.savestates = np.array([np.copy(self.masks) for _ in range(10)])
        self.resize_image()
        self.window_update()
    """à faire: resize image, fait mais à modifier pour prendre mask aussi, cette fct devrait juste renvoyer les valeurs de tailles et appeler 
    - image + mask display, appelée au debut, en modif de mask, en resize"""







        

if __name__ == "__main__":
    #SAM<
    print("Loading Segment Anything Model...", end="\t")
    import warnings
    onnx_model_path = "sam_onnx_example.onnx"
    onnx_model = SamOnnxModel(sam, return_single_mask=True)
    dynamic_axes = {
        "point_coords": {1: "num_points"},"point_labels": {1: "num_points"},
    }
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),"point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),"mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),"orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,tuple(dummy_inputs.values()),f,export_params=True,verbose=False,opset_version=17,
                do_constant_folding=True,input_names=list(dummy_inputs.keys()),output_names=output_names,dynamic_axes=dynamic_axes,
            )
    print("...SAM loaded !")
    #>SAM

    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
    