import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting
import numpy

class Seeding(cellprofiler.module.ImageProcessing):
    module_name = "Seeding"
    variable_revision_number = 1

    def create_settings(self):
        super(Seeding, self).create_settings()

    def settings(self):
        settings = super(Seeding, self).settings()
        return settings

    def visible_settings(self):
        visible_settings = super(Seeding, self).visible_settings()
        return visible_settings

    def run(self, workspace):
        x_name = self.x_name.value
        images = workspace.image_set
        x = images.get_image(x_name)
        dimensions = x.dimensions
        y_data = x.pixel_data.copy()
        floor_object_number = numpy.amax(y_data) + 1
        ceiling_object_number = floor_object_number + 1
        y_data[0, :, :] = floor_object_number
        y_data[-1, :, :] = ceiling_object_number
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