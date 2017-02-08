import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage.io
import skimage.morphology
import skimage.util

class AdaptiveLocalNormalization(cellprofiler.module.ImageProcessing):
    module_name = "AdaptiveLocalNormalization"
    variable_revision_number = 1

    def aln_normalize(self, img, pixel_location, radius, T_StD):
        """
        :param self: an instance of AdaptiveLocalNormalization class
        :param img: a 2D ndarray/image
        :param pixel_location: a tuple with 2 elements (row, col), coordinates in an ndarray
        :param radius: radius of a disc structuring element that defines neighborhood of normalization
        :param T_StD: the standard deviation threshold. A fraction of the image std.
        :return I_N: The normalized value at the pixel_location
        """
        xy_region, pad_width = self.xy_region_and_pad_width(img.shape, pixel_location, radius)
        if pad_width > 0:
            img_pad = skimage.util.pad(img, pad_width, "symmetric")
        else:
            img_pad = img
        img_mask = numpy.zeros_like(img_pad)
        strel = skimage.morphology.disk(radius)
        img_mask[xy_region[0]:xy_region[1], xy_region[2]:xy_region[3]] = strel
        I_E = numpy.mean(img_pad[numpy.nonzero(img_mask)])
        I_StD = numpy.std(img_pad[numpy.nonzero(img_mask)])
        # to prevent low variance regions, compare I_StD to T_StD
        I_StD = T_StD if I_StD < T_StD else I_StD
        I_N = (img[pixel_location] - I_E) / I_StD
        return I_N

    def calculate_aln_image(self, img, radius_image, threshold_std):
        """
        :param self: an instance of AdaptiveLocalNormalization class
        :param img: a 2D ndarray/image
        :param radius_image: a 2D ndarray the same size as img with search_radius for every pixel
        :param threshold_std: a float within [0,1.0). It specifies a fraction of the global_std.
        :return aln_image: img after normalization. A 2D ndarray the same size as img.
        """
        T_StD = threshold_std * numpy.std(img)
        aln_image = numpy.zeros_like(img)
        indexiter = numpy.nditer(img, flags=['multi_index'])
        while not indexiter.finished:
            pixel_location = indexiter.multi_index
            # do stuff zone
            aln_pixel = self.aln_normalize(img, pixel_location, radius_image[pixel_location].astype("int"), T_StD)
            aln_image[pixel_location] = aln_pixel
            indexiter.iternext()
        return aln_image

    def calculate_radius_image(self, img, threshold_std, radius_max_float):
        """
        :param self: an instance of AdaptiveLocalNormalization class
        :param img: a 2D ndarray/image
        :param threshold_std: a float within [0,1.0). It specifies a fraction of the global_std.
        :return radius_image: a 2D ndarray the same size as img with search_radius for every pixel
        """
        radius_max = numpy.mean(img.shape).astype("int")
        radius_image = numpy.zeros_like(img)
        global_std = numpy.std(img)
        indexiter = numpy.nditer(img, flags=['multi_index'])
        while not indexiter.finished:
            pixel_location = indexiter.multi_index
            # do stuff zone
            aln_radius = self.find_aln_radius(img, pixel_location, global_std, threshold_std, radius_max)
            radius_image[pixel_location] = aln_radius
            indexiter.iternext()
        return radius_image

    def create_settings(self):
        super(AdaptiveLocalNormalization, self).create_settings()

        self.threshold_std_float = cellprofiler.setting.Float(
            "Standard Deviation Threshold", 0.5, doc="""
                    Enter a number within the range [0,1.0).
                    This specifies a fraction of the standard deviation of the image.""")

        self.radius_max_float = cellprofiler.setting.Float(
            "Maximum Radius", 0.5, doc="""
                    Enter a number within the range (0,1.0].
                    This specifies the largest radius allowed during the radius search for each pixel.""")

    def find_aln_radius(self, img, pixel_location, global_std, threshold_std, radius_max):
        """
        :param self: an instance of AdaptiveLocalNormalization class
        :param img: a 2D ndarray/image
        :param pixel_location: a tuple with 2 elements (row, col), coordinates in an ndarray
        :param global_std: the standard deviation of img, pre-calculated for speed since this function will be called for every pixel
        :param threshold_std: a float within [0,1.0). It specifies a fraction of the global_std.
        :param radius_max: an integer that will stop the search when the radius reaches this number
        :return search_radius: an integer > 0. The radius of a disc structuring element.
        """
        T_StD = global_std * threshold_std
        # prime the while loop...
        search_condition = True
        # create mask
        search_radius = 1
        xy_region, pad_width = self.xy_region_and_pad_width(img.shape, pixel_location, search_radius)
        if pad_width > 0:
            img_pad = skimage.util.pad(img, pad_width, "symmetric")
        else:
            img_pad = img
        img_mask = numpy.zeros(img_pad.shape)
        strel = skimage.morphology.disk(search_radius)
        img_mask[xy_region[0]:xy_region[1], xy_region[2]:xy_region[3]] = strel
        #
        while search_condition:
            # calculate local std
            search_std = numpy.std(img_pad[numpy.nonzero(img_mask)])
            if search_std > T_StD:
                search_condition = False
                break
            # search_std is < cutoff, so search continues
            # create mask
            search_radius = search_radius + 1
            xy_region, pad_width = self.xy_region_and_pad_width(img.shape, pixel_location, search_radius)
            if pad_width > 0:
                img_pad = skimage.util.pad(img, pad_width, "symmetric")
            else:
                img_pad = img
            img_mask = numpy.zeros(img_pad.shape)
            strel = skimage.morphology.disk(search_radius)
            img_mask[xy_region[0]:xy_region[1], xy_region[2]:xy_region[3]] = strel
            # check for radius_max
            if search_radius >= radius_max:
                search_condition = False
                break
        return search_radius

    def settings(self):
        __settings__ = super(AdaptiveLocalNormalization, self).settings()

        return __settings__ + [
            self.threshold_std_float,
            self.radius_max_float
        ]

    def visible_settings(self):
        __settings__ = super(AdaptiveLocalNormalization, self).settings()

        return __settings__ + [
            self.threshold_std_float,
            self.radius_max_float
        ]

    def run(self, workspace):
        x_name = self.x_name.value
        images = workspace.image_set
        x = images.get_image(x_name)
        dimensions = x.dimensions
        x_data = x.pixel_data.copy()
        # perform normalization slice by slice for z-stack
        y_data = numpy.zeros_like(x_data)
        # The if-clause determines if data is 2D or 3D
        if len(x_data.shape) > 2:
            for index, plane in enumerate(x_data):
                radius_image = self.calculate_radius_image(plane, self.threshold_std_float.value,
                                                           self.radius_max_float.value)
                y_data[index] = self.calculate_aln_image(plane, radius_image, self.threshold_std_float.value)
        else:
            radius_image = self.calculate_radius_image(x_data, self.threshold_std_float.value,
                                                       self.radius_max_float.value)
            y_data = self.calculate_aln_image(x_data, radius_image, self.threshold_std_float.value)
        y = cellprofiler.image.Image(
            dimensions=dimensions,
            image=y_data,
            parent_image=x
        )
        y_name = self.y_name.value
        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x.pixel_data

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

    def xy_region_and_pad_width(self, img_shape, pixel_location, search_radius):
        """
        :param self: an instance of AdaptiveLocalNormalization class
        :param img_shape: A tuple with 2 elements (rows, cols) of an ndarray/image
        :param pixel_location: a tuple with 2 elements (row, col), coordinates in an ndarray
        :param search_radius: an integer > 0. The radius of a disc structuring element.
        :return xy_region: a tuple with 4 elements describing the range of rows and cols that holds the strel.
        (row0,row1,col0,col1)
        :return pad_width: the number of pixels required to pad an image, so that a strel can fit within it.
        """
        pad_width_width_upperleft = pixel_location[0] - search_radius
        pad_width_height_upperleft = pixel_location[1] - search_radius
        pad_width_upperleft = numpy.abs(numpy.amin([pad_width_width_upperleft, pad_width_height_upperleft]))
        pad_width_width_lowerright = pixel_location[0] + search_radius + 1 - img_shape[0]
        pad_width_height_lowerright = pixel_location[1] + search_radius + 1 - img_shape[1]
        pad_width_lowerright = numpy.amax([pad_width_width_lowerright, pad_width_height_lowerright])
        pad_width = numpy.amax([pad_width_upperleft, pad_width_lowerright])

        xy_region = (pixel_location[0] - search_radius + pad_width,
                     pixel_location[0] + search_radius + 1 + pad_width,
                     pixel_location[1] - search_radius + pad_width,
                     pixel_location[1] + search_radius + 1 + pad_width)

        return xy_region, pad_width


