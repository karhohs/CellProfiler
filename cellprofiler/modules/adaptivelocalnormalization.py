import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy
import skimage
import skimage.io
import skimage.morphology
import skimage.util
import scipy.ndimage.filters
import skimage.exposure


class AdaptiveLocalNormalization(cellprofiler.module.ImageProcessing):
    """
    Things to keep in mind: There is a fast version and a slow version of normalization. The slow version is true to
    the algorithm put forth in the paper. However, using disc structuring elements with scipy generic filtering is
    time intensive. An alternative is to use scipy uniform filtering. It is much faster, but it only works in
    rectangular windows. The trade-off for speed can be useful, so both slow and fast have been included.

    The normalization produces both positive and negative float values. It is preferred to use uint16 ndarrays,
    so after calculation of the image it is scaled to fit within the uint16 format. This might introduce some
    unwanted variation if stitching images or processing a group of images, because each is scaled independently.
    However, if the distribution of intensities across images is similar then this shouldn't too costly. To globally
    scale a set of images would require more coding overhead...
    """
    module_name = "AdaptiveLocalNormalization"
    variable_revision_number = 1

    def create_settings(self):
        super(AdaptiveLocalNormalization, self).create_settings()

        self.threshold_std_float = cellprofiler.setting.Float(
            "Standard Deviation Threshold",
            0.5,
            doc="""
            Enter a number within the range [0,1.0). This specifies a fraction of the standard deviation of the image.
            """
        )

        self.radius_max = cellprofiler.setting.Integer(
            "Maximum Radius",
            25,
            doc="""
            Enter an integer greater than 0. This specifies the largest radius allowed during the radius search for
            each pixel. The larger the radius the longer the run time.
            """
        )

        self.radius_method = cellprofiler.setting.Choice(
            "Radius Method",
            ["cv", "std"],
            doc="""
            *cv* uses the coefficient of variation to calculate the radius.
            *std* uses the standard deviation to calculate the radius.
            """,
            value="std"
        )

        self.strel_shape = cellprofiler.setting.Choice(
            "Structuring Element Shape",
            ["square", "disc"],
            doc="""
            *square* is approx. 1000 times faster than *disc*.
            *disc* is true to the source algorithm.
            """,
            value="square"
        )

    def settings(self):
        __settings__ = super(AdaptiveLocalNormalization, self).settings()

        return __settings__ + [
            self.threshold_std_float,
            self.radius_max,
            self.radius_method,
            self.strel_shape
        ]

    def uniform_filter_std(self, img, window_size):
        """
        :param self: an instance of AdaptiveLocalNormalization class
        :param img: a 2D ndarray/image
        :param window_size: a square array is used to filter img. window_size defines the size of the array, e.g. (n,n)
        :return img_std: the img after applying a standard deviation filter.

        Reference:
        https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html

        The scipy function will return nan in unexpected ways, so they are removed before returning the img.
        """
        c1 = scipy.ndimage.filters.uniform_filter(img, window_size)
        c2 = scipy.ndimage.filters.uniform_filter(numpy.multiply(img, img), window_size)
        img_std = numpy.sqrt(c2 - numpy.multiply(c1, c1))
        img_std[numpy.isnan(img_std)] = numpy.mean(img_std)
        return img_std

    def visible_settings(self):
        __settings__ = super(AdaptiveLocalNormalization, self).settings()

        return __settings__ + [
            self.threshold_std_float,
            self.radius_max,
            self.radius_method,
            self.strel_shape
        ]

    def run(self, workspace):
        x_name = self.x_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data = skimage.img_as_float(x.pixel_data)

        # The if-clause determines if data is 2D or 3D
        if x.volumetric:
            # perform normalization slice by slice for z-stack
            y_data = numpy.zeros_like(x_data)

            for index, plane in enumerate(x_data):
                radius_image = self.calculate_radius_image(
                    plane,
                    self.threshold_std_float.value,
                    self.radius_max.value
                )

                y_data[index] = self.calculate_aln_image(
                    plane,
                    radius_image,
                    self.threshold_std_float.value,
                    self.radius_max.value
                )
        else:
            radius_image = self.calculate_radius_image(
                x_data,
                self.threshold_std_float.value,
                self.radius_max.value
            )

            y_data = self.calculate_aln_image(
                x_data,
                radius_image,
                self.threshold_std_float.value,
                self.radius_max.value
            )

        y_data = skimage.img_as_float(y_data)

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

    def calculate_aln_image(self, img, radius_image, threshold_std, radius_max):
        """
        :param self: an instance of AdaptiveLocalNormalization class
        :param img: a 2D ndarray/image
        :param radius_image: a 2D ndarray the same size as img with search_radius for every pixel
        :param threshold_std: a float within [0,1.0). It specifies a fraction of the global_std.
        :param radius_max: The largest radius allowed during the radius search for each pixel
        :return aln_image: img after normalization. A 2D ndarray the same size as img.
        """
        aln_image = numpy.zeros_like(img)

        global_std = numpy.std(img)

        global_mean = numpy.mean(img)

        t_std = global_std * threshold_std

        t_cv = t_std / global_mean

        radii = numpy.arange(radius_max) + 1

        for r in radii:
            if self.strel_shape.value == "disc":
                if self.radius_method.value == "std":
                    strel = skimage.morphology.disk(r)

                    img_std = scipy.ndimage.filters.generic_filter(img, numpy.std, footprint=strel)

                    img_std[img_std < t_std] = t_std

                    img_mean = scipy.ndimage.filters.generic_filter(img, numpy.mean, footprint=strel)

                    img_norm = (img - img_mean) / img_std

                    aln_image[radius_image == r] = img_norm[radius_image == r]
                else:
                    strel = skimage.morphology.disk(r)

                    img_std = scipy.ndimage.filters.generic_filter(img, numpy.std, footprint=strel)

                    img_mean = scipy.ndimage.filters.generic_filter(img, numpy.mean, footprint=strel)

                    img_std[img_std < t_std] = t_std

                    img_norm = (img - img_mean) / img_std

                    aln_image[radius_image == r] = img_norm[radius_image == r]
            else:
                if self.radius_method.value == "std":
                    img_std = self.uniform_filter_std(img, r)

                    img_std[img_std < t_std] = t_std

                    img_mean = self.uniform_filter_mean(img, r)

                    img_norm = (img - img_mean) / img_std

                    aln_image[radius_image == r] = img_norm[radius_image == r]
                else:
                    img_std = self.uniform_filter_std(img, r)

                    img_mean = self.uniform_filter_mean(img, r)

                    img_std[img_std < t_std] = t_std

                    img_norm = (img - img_mean) / img_std

                    aln_image[radius_image == r] = img_norm[radius_image == r]

        return aln_image

    def calculate_radius_image(self, img, threshold_std, radius_max):
        """
        :param self: an instance of AdaptiveLocalNormalization class
        :param img: a 2D ndarray/image
        :param threshold_std: a float within [0,1.0). It specifies a fraction of the global_std.
        :param radius_max: The largest radius allowed during the radius search for each pixel
        :return radius_image: a 2D ndarray the same size as img with search_radius for every pixel
        """
        radius_image = numpy.zeros_like(img)

        global_std = numpy.std(img)

        global_mean = numpy.mean(img)

        t_std = global_std * threshold_std

        t_cv = t_std / global_mean

        radii = numpy.arange(radius_max) + 1

        for r in radii:
            if self.strel_shape.value == "disc":
                if self.radius_method.value == "std":
                    strel = skimage.morphology.disk(r)

                    img_std = scipy.ndimage.filters.generic_filter(img, numpy.std, footprint=strel)

                    img_bool = img_std <= t_std

                    radius_image[img_bool] = r
                else:
                    strel = skimage.morphology.disk(r)

                    img_std = scipy.ndimage.filters.generic_filter(img, numpy.std, footprint=strel)

                    img_mean = scipy.ndimage.filters.generic_filter(img, numpy.mean, footprint=strel)

                    img_cv = img_std / (img_mean + 1)  # the +1 is okay if uint8 or uint16

                    img_bool = img_cv <= t_cv

                    radius_image[img_bool] = r
            else:
                if self.radius_method.value == "std":
                    img_std = self.uniform_filter_std(img, r)

                    img_bool = img_std <= t_std

                    radius_image[img_bool] = r
                else:
                    img_std = self.uniform_filter_std(img, r)

                    img_mean = self.uniform_filter_mean(img, r)  # remove

                    img_cv = img_std / (img_mean + 1)  # the +1 is okay if uint8 or uint16

                    img_bool = img_cv <= t_cv

                    radius_image[img_bool] = r

        return radius_image

    def uniform_filter_mean(self, img, window_size):
        """
        :param self: an instance of AdaptiveLocalNormalization class
        :param img: a 2D ndarray/image
        :param window_size: a square array is used to filter img. window_size defines the size of the array, e.g. (n,n)
        :return img_mean: the img after applying a mean filter.

        Reference:
        https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html

        The scipy function will return nan in unexpected ways, so they are removed before returning the img.
        """
        img_mean = scipy.ndimage.filters.uniform_filter(img, window_size)

        img_mean[numpy.isnan(img_mean)] = numpy.mean(img_mean)

        return img_mean