#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import unittest
import tempfile
import os
from PIL import Image
import numpy as np
#Internal libraries
from aucmedi import *
from aucmedi.xai import *
from aucmedi.xai.methods import *
from aucmedi.data_processing.io_loader import image_loader, numpy_loader
#-----------------------------------------------------#
#              Unittest: Explainable AI               #
#-----------------------------------------------------#
class xaiTEST(unittest.TestCase):
    # Setup AUCMEDI pipeline
    @classmethod
    def setUpClass(self):
        np.random.seed(1234)
        # Initialize temporary directory
        self.tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                                    suffix=".data")
        # Create RGB data
        self.sampleList = []
        for i in range(0, 10):
            img_rgb = np.random.rand(32, 32, 3) * 255
            img_pillow = Image.fromarray(img_rgb.astype(np.uint8))
            index = "image.sample_" + str(i) + ".RGB.png"
            path_sample = os.path.join(self.tmp_data.name, index)
            img_pillow.save(path_sample)
            self.sampleList.append(index)
        # Create classification labels
        self.labels_ohe = np.zeros((10, 4), dtype=np.uint8)
        for i in range(0, 10):
            class_index = np.random.randint(0, 4)
            self.labels_ohe[i][class_index] = 1

        # Create Data Generator
        self.datagen = DataGenerator(self.sampleList,  self.tmp_data.name,
                                     labels=self.labels_ohe, resize=None,
                                     standardize_mode=None,
                                     grayscale=False, batch_size=3)
        # Create Neural Network model
        self.model = NeuralNetwork(n_labels=4, channels=3, input_shape=(32,32),
                                    architecture="2D.Vanilla")
        # Compute predictions
        self.preds = self.model.predict(self.datagen)
        # Initialize testing image
        self.image = self.datagen[0][0][[0]]

        # ----------- 3D TestData -----------
        self.sampleList_hu_3D = []
        self.sampleList_rgb_3D = []
        for i in range(0, 10):
            img_hu_3D = (np.random.rand(32, 32, 32) * 2000) - 500
            img_rgb_3D = np.random.rand(32, 32, 32, 3) * 255

            path_hu = os.path.join(self.tmp_data.name, f"vol_hu_{i}.npy")
            np.save(path_hu, img_hu_3D)
            self.sampleList_hu_3D.append(f"vol_hu_{i}.npy")

            path_rgb = os.path.join(self.tmp_data.name, f"vol_rgb_{i}.npy")
            np.save(path_rgb, img_rgb_3D)
            self.sampleList_rgb_3D.append(f"vol_rgb_{i}.npy")

        self.labels_ohe_hu_3D = np.zeros((10, 4), dtype=np.uint8)
        self.labels_ohe_rgb_3D = np.zeros((10, 4), dtype=np.uint8)
        for i in range(10):
            class_index = np.random.randint(0, 4)
            self.labels_ohe_hu_3D[i][class_index] = 1
            self.labels_ohe_rgb_3D[i][class_index] = 1

        self.datagen_hu_3D = DataGenerator(self.sampleList_hu_3D, self.tmp_data.name,
                                        labels=self.labels_ohe_hu_3D,
                                        loader=numpy_loader,
                                        two_dim=False,
                                        resize=(32, 32, 32), standardize_mode=None,
                                        grayscale=True, batch_size=3)
        
        self.datagen_rgb_3D = DataGenerator(self.sampleList_rgb_3D, self.tmp_data.name,
                                        labels=self.labels_ohe_rgb_3D, 
                                        loader=numpy_loader,
                                        two_dim=False,
                                        resize=(32, 32, 32), standardize_mode=None,
                                        grayscale=False, batch_size=3)

        self.model_hu_3D = NeuralNetwork(n_labels=4, channels=1,
                                      input_shape=(32, 32, 32),
                                      architecture="3D.Vanilla")
        
        self.model_rgb_3D = NeuralNetwork(n_labels=4, channels=3,
                                      input_shape=(32, 32, 32),
                                      architecture="3D.Vanilla")
        
        self.preds_hu_3D = self.model_hu_3D.predict(self.datagen_hu_3D)
        self.image_hu_3D = self.datagen_hu_3D[0][0][[0]]

        self.preds_rgb_3D = self.model_rgb_3D.predict(self.datagen_rgb_3D)
        self.image_rgb_3D = self.datagen_rgb_3D[0][0][[0]]

    #-------------------------------------------------#
    #             XAI Functions: Decoder              #
    #-------------------------------------------------#
    def test_Decoder_argmax_output(self):
        imgs, hms = xai_decoder(self.datagen, self.model, preds=self.preds,
                                out_path=None)
        self.assertTrue(np.array_equal(np.array(imgs).shape, (10, 32, 32, 3)))
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 32, 32)))

    def test_Decoder_argmax_visualize(self):
        path_xai = os.path.join(self.tmp_data.name, "xai")
        xai_decoder(self.datagen, self.model, preds=self.preds, out_path=path_xai)
        for i in range(0, len(self.sampleList)):
            path_xai_file = os.path.join(os.path.join(self.tmp_data.name, "xai"),
                                         self.sampleList[i])
            self.assertTrue(os.path.exists(path_xai_file))
            img = image_loader(sample=self.sampleList[i],
                               path_imagedir=self.tmp_data.name,
                               image_format=self.datagen.image_format)
            hm = image_loader(sample=self.sampleList[i],
                              path_imagedir=os.path.join(self.tmp_data.name, "xai"),
                              image_format=self.datagen.image_format)
            self.assertTrue(np.array_equal(img.shape, hm.shape))
            self.assertFalse(np.array_equal(img, hm))

    def test_Decoder_allclasses_output(self):
        imgs, hms = xai_decoder(self.datagen, self.model, preds=None,
                                out_path=None)
        self.assertTrue(np.array_equal(np.array(imgs).shape, (10, 32, 32, 3)))
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))

    def test_Decoder_allclasses_visualize(self):
        path_xai = os.path.join(self.tmp_data.name, "xai")
        xai_decoder(self.datagen, self.model, preds=None, out_path=path_xai)
        for i in range(0, len(self.sampleList)):
            sample = self.sampleList[i]
            for c in range(0, 4):
                xai_file = sample[:-4] + ".class_" + str(c) + sample[-4:]
                img = image_loader(sample=self.sampleList[i],
                                   path_imagedir=self.tmp_data.name,
                                   image_format=self.datagen.image_format)
                hm = image_loader(sample=xai_file,
                                  path_imagedir=os.path.join(self.tmp_data.name, "xai"),
                                  image_format=self.datagen.image_format)
                self.assertTrue(np.array_equal(img.shape, hm.shape))
                self.assertFalse(np.array_equal(img, hm))

    def test_Decoder_directoryInterface(self):
        # Create imaging data with subdirectories
        tmp_data = tempfile.TemporaryDirectory(prefix="tmp.aucmedi.",
                                               suffix=".data")
        for i in range(0, 5):
            os.mkdir(os.path.join(tmp_data.name, "class_" + str(i)))
        # Fill subdirectories with images
        for i in range(0, 25):
            img = np.random.rand(32, 32, 3) * 255
            img_pillow = Image.fromarray(img.astype(np.uint8))
            index = "image.sample_" + str(i) + ".png"
            label_dir = "class_" + str((i % 5))
            path_sample = os.path.join(tmp_data.name, label_dir, index)
            img_pillow.save(path_sample)
        ds = input_interface(interface="directory", path_imagedir=tmp_data.name)
        (index_list, _, nclasses, _, _) = ds
        # Create Data Generator
        datagen = DataGenerator(index_list,  tmp_data.name,
                                labels=None, resize=None,
                                grayscale=False, batch_size=3)
        # Create Neural Network model
        model = NeuralNetwork(n_labels=nclasses, channels=3, input_shape=(32,32),
                              architecture="2D.Vanilla")
        path_xai = os.path.join(tmp_data.name, "xai_directory")
        xai_decoder(datagen, model, preds=None, out_path=path_xai)

    #-------------------------------------------------#
    #              XAI Methods: Grad-Cam              #
    #-------------------------------------------------#
    def test_XAImethod_GradCam_init(self):
        GradCAM(self.model.model)
        xai_dict["gradcam"](self.model.model)
        xai_dict["gc"](self.model.model)

    def test_XAImethod_GradCam_heatmap(self):
        xai_method = GradCAM(self.model.model)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (2,2)))

    def test_XAImethod_GradCam3D_heatmap(self):
        xai_method_hu = GradCAM(self.model_hu_3D.model)
        xai_method_rgb = GradCAM(self.model_rgb_3D.model)
        for i in range(4):
            hm_hu = xai_method_hu.compute_heatmap(image=self.image_hu_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_hu.shape, (2, 2, 2)))

            hm_rgb = xai_method_rgb.compute_heatmap(image=self.image_rgb_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_rgb.shape, (2, 2, 2)))

    def test_XAImethod_GradCam_decoder(self):
        imgs, hms = xai_decoder(self.datagen, self.model, method="gradcam")
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))

    def test_XAImethod_GradCam3D_decoder(self):
        imgs_hu, hms_hu = xai_decoder(self.datagen_hu_3D, self.model_hu_3D, method="gradcam")
        self.assertTrue(np.array_equal(np.array(hms_hu).shape, (10, 4, 32, 32, 32)))

        imgs_rgb, hms_rgb = xai_decoder(self.datagen_rgb_3D, self.model_rgb_3D, method="gradcam")
        self.assertTrue(np.array_equal(np.array(hms_rgb).shape, (10, 4, 32, 32, 32)))

    #-------------------------------------------------#
    #             XAI Methods: Grad-Cam++             #
    #-------------------------------------------------#
    def test_XAImethod_GradCamPP_init(self):
        GradCAMpp(self.model.model)
        xai_dict["gradcam++"](self.model.model)
        xai_dict["gc++"](self.model.model)

    def test_XAImethod_GradCamPP_heatmap(self):
        xai_method = GradCAMpp(self.model.model)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (2,2)))

    def test_XAImethod_GradCamPP3D_heatmap(self):
        xai_method_hu = GradCAMpp(self.model_hu_3D.model)
        xai_method_rgb = GradCAMpp(self.model_rgb_3D.model)
        for i in range(4):
            hm_hu = xai_method_hu.compute_heatmap(image=self.image_hu_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_hu.shape, (2, 2, 2)))

            hm_rgb = xai_method_rgb.compute_heatmap(image=self.image_rgb_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_rgb.shape, (2, 2, 2)))

    def test_XAImethod_GradCamPP_decoder(self):
        imgs, hms = xai_decoder(self.datagen, self.model, method="gradcam++")
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))

    def test_XAImethod_GradCamPP3D_decoder(self):
        imgs_hu, hms_hu = xai_decoder(self.datagen_hu_3D, self.model_hu_3D, method="gradcam++")
        self.assertTrue(np.array_equal(np.array(hms_hu).shape, (10, 4, 32, 32, 32)))

        imgs_rgb, hms_rgb = xai_decoder(self.datagen_rgb_3D, self.model_rgb_3D, method="gradcam++")
        self.assertTrue(np.array_equal(np.array(hms_rgb).shape, (10, 4, 32, 32, 32)))



    #-------------------------------------------------#
    #            XAI Methods: Saliency Maps           #
    #-------------------------------------------------#
    def test_XAImethod_SaliencyMap_init(self):
        SaliencyMap(self.model.model)
        xai_dict["saliency"](self.model.model)
        xai_dict["sm"](self.model.model)

    def test_XAImethod_SaliencyMap_heatmap(self):
        xai_method = SaliencyMap(self.model.model)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_SaliencyMap3D_heatmap(self):
        xai_method_hu = SaliencyMap(self.model_hu_3D.model)
        xai_method_rgb = SaliencyMap(self.model_rgb_3D.model)
        for i in range(4):
            hm_hu = xai_method_hu.compute_heatmap(image=self.image_hu_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_hu.shape, (32, 32, 32)))

            hm_rgb = xai_method_rgb.compute_heatmap(image=self.image_rgb_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_rgb.shape, (32, 32, 32)))

    def test_XAImethod_SaliencyMap_decoder(self):
        imgs, hms = xai_decoder(self.datagen, self.model, method="saliency")
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))

    def test_XAImethod_SaliencyMap3D_decoder(self):
        imgs_hu, hms_hu = xai_decoder(self.datagen_hu_3D, self.model_hu_3D, method="saliency")
        self.assertTrue(np.array_equal(np.array(hms_hu).shape, (10, 4, 32, 32, 32)))

        imgs_rgb, hms_rgb = xai_decoder(self.datagen_rgb_3D, self.model_rgb_3D, method="saliency")
        self.assertTrue(np.array_equal(np.array(hms_rgb).shape, (10, 4, 32, 32, 32)))

    #-------------------------------------------------#
    #       XAI Methods: Guided Backpropagation       #
    #-------------------------------------------------#
    def test_XAImethod_GuidedBackprop_init(self):
        GuidedBackpropagation(self.model.model)
        xai_dict["guidedbackprop"](self.model.model)
        xai_dict["gb"](self.model.model)

    def test_XAImethod_GuidedBackprop_heatmap(self):
        xai_method = GuidedBackpropagation(self.model.model)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_GuidedBackprop3D_heatmap(self):
        xai_method_hu = GuidedBackpropagation(self.model_hu_3D.model)
        xai_method_rgb = GuidedBackpropagation(self.model_rgb_3D.model)
        for i in range(4):
            hm_hu = xai_method_hu.compute_heatmap(image=self.image_hu_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_hu.shape, (32, 32, 32)))

            hm_rgb = xai_method_rgb.compute_heatmap(image=self.image_rgb_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_rgb.shape, (32, 32, 32)))

    def test_XAImethod_GuidedBackprop_decoder(self):
        imgs, hms = xai_decoder(self.datagen, self.model, method="guidedbackprop")
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))

    def test_XAImethod_GuidedBackprop3D_decoder(self):
        imgs_hu, hms_hu = xai_decoder(self.datagen_hu_3D, self.model_hu_3D, method="guidedbackprop")
        self.assertTrue(np.array_equal(np.array(hms_hu).shape, (10, 4, 32, 32, 32)))

        imgs_rgb, hms_rgb = xai_decoder(self.datagen_rgb_3D, self.model_rgb_3D, method="guidedbackprop")
        self.assertTrue(np.array_equal(np.array(hms_rgb).shape, (10, 4, 32, 32, 32)))

    #-------------------------------------------------#
    #        XAI Methods: Integrated Gradients        #
    #-------------------------------------------------#
    def test_XAImethod_IntegratedGradients_init(self):
        IntegratedGradients(self.model.model)
        xai_dict["IntegratedGradients"](self.model.model)
        xai_dict["ig"](self.model.model)

    def test_XAImethod_IntegratedGradients_heatmap(self):
        xai_method = IntegratedGradients(self.model.model)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_IntegratedGradients_decoder(self):
        imgs, hms = xai_decoder(self.datagen, self.model, method="IntegratedGradients")
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))

    #-------------------------------------------------#
    #           XAI Methods: Guided Grad-CAM          #
    #-------------------------------------------------#
    def test_XAImethod_GuidedGradCAM_init(self):
        GuidedGradCAM(self.model.model)
        xai_dict["GuidedGradCAM"](self.model.model)
        xai_dict["ggc"](self.model.model)

    def test_XAImethod_GuidedGradCAM_heatmap(self):
        xai_method = GuidedGradCAM(self.model.model)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_GuidedGradCAM3D_heatmap(self):
        xai_method_hu = GuidedGradCAM(self.model_hu_3D.model)
        xai_method_rgb = GuidedGradCAM(self.model_rgb_3D.model)
        for i in range(4):
            hm_hu = xai_method_hu.compute_heatmap(image=self.image_hu_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_hu.shape, (32, 32, 32)))

            hm_rgb = xai_method_rgb.compute_heatmap(image=self.image_rgb_3D, class_index=i)
            self.assertTrue(np.array_equal(hm_rgb.shape, (32, 32, 32)))

    def test_XAImethod_GuidedGradCAM_decoder(self):
        imgs, hms = xai_decoder(self.datagen, self.model, method="GuidedGradCAM")
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))

    def test_XAImethod_GuidedGradCAM3D_decoder(self):
        imgs_hu, hms_hu = xai_decoder(self.datagen_hu_3D, self.model_hu_3D, method="GuidedGradCAM")
        self.assertTrue(np.array_equal(np.array(hms_hu).shape, (10, 4, 32, 32, 32)))

        imgs_rgb, hms_rgb = xai_decoder(self.datagen_rgb_3D, self.model_rgb_3D, method="GuidedGradCAM")
        self.assertTrue(np.array_equal(np.array(hms_rgb).shape, (10, 4, 32, 32, 32)))

    #-------------------------------------------------#
    #        XAI Methods: Occlusion Sensitivity       #
    #-------------------------------------------------#
    def test_XAImethod_OcclusionSensitivity_init(self):
        OcclusionSensitivity(self.model.model, patch_size=16)
        xai_dict["OcclusionSensitivity"](self.model.model)
        xai_dict["os"](self.model.model)

    def test_XAImethod_OcclusionSensitivity_heatmap(self):
        xai_method = OcclusionSensitivity(self.model.model, patch_size=16)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_OcclusionSensitivity_decoder(self):
        imgs, hms = xai_decoder(self.datagen, self.model, method="OcclusionSensitivity")
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))

    #-------------------------------------------------#
    #              XAI Methods: LIME Con              #
    #-------------------------------------------------#
    def test_XAImethod_LimeCon_init(self):
        LimeCon(self.model.model, num_samples=10)
        xai_dict["LimeCon"](self.model.model)
        xai_dict["lc"](self.model.model)

    def test_XAImethod_LimeCon_heatmap(self):
        xai_method = LimeCon(self.model.model, num_samples=10)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_LimeCon_decoder(self):
        xai_method = LimeCon(self.model.model, num_samples=10)
        imgs, hms = xai_decoder(self.datagen, self.model, method=xai_method)
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))

    #-------------------------------------------------#
    #              XAI Methods: LIME Pro              #
    #-------------------------------------------------#
    def test_XAImethod_LimePro_init(self):
        LimePro(self.model.model, num_samples=10)
        xai_dict["LimePro"](self.model.model)
        xai_dict["lp"](self.model.model)

    def test_XAImethod_LimePro_heatmap(self):
        xai_method = LimePro(self.model.model, num_samples=10)
        for i in range(4):
            hm = xai_method.compute_heatmap(image=self.image, class_index=i)
            self.assertTrue(np.array_equal(hm.shape, (32,32)))

    def test_XAImethod_LimePro_decoder(self):
        xai_method = LimePro(self.model.model, num_samples=10)
        imgs, hms = xai_decoder(self.datagen, self.model, method=xai_method)
        self.assertTrue(np.array_equal(np.array(hms).shape, (10, 4, 32, 32)))
