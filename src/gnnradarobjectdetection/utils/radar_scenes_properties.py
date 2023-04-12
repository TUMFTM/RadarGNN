
class Labels:
    """ Holds the definition of the label ID and the corresponding class (same definition as proposed by RadarScenes)
    """
    @staticmethod
    def get_label_dict():

        label_dict = {0: "car",
                      1: "pedestrian",
                      2: "pedestrian group",
                      3: "two wheeler",
                      4: "large vehicle",
                      5: "background"}

        return label_dict


class Colors:
    """ Helper class for colors. Provides mappings from sensor ids to colors as well as colors for the individual classes.
    """

    red = "#f02b2b"
    blue = "#4763ff"
    green = "#47ff69"
    light_green = "#73ff98"
    orange = "#ff962e"
    violet = "#c561d4"
    indigo = "#8695e3"
    grey = "#7f8c8d"
    yellow = "#ffff33"
    lime = "#c6ff00"
    amber = "#ffd54f"
    teal = "#19ffd2"
    pink = "#ff6eba"
    brown = "#c97240"
    black = "#1e272e"
    midnight_blue = "#34495e"
    deep_orange = "#e64a19"
    light_blue = "#91cded"
    light_gray = "#dedede"
    gray = "#888888"

    sensor_id_to_color = {
        1: red,
        2: blue,
        3: green,
        4: pink
    }

    label_id_to_color = {
        0: violet,
        1: orange,
        2: green,
        3: pink,
        4: light_blue,
        5: gray,
        6: brown,
        7: yellow,
        8: light_green,
        9: blue,
        10: indigo,
        11: teal
    }

    object_colors = [red, blue, green, light_green, orange, violet, yellow, teal, pink, brown,
                     light_blue, lime, deep_orange, amber, indigo]


class ClassDistribution:

    @staticmethod
    def get_radar_point_dict():
        """ Describes number of points corresponding to the classes of the RadarScenes dataset
        """
        radar_point_dict = {"car": 2.1e6,
                            "pedestrian": 5.1e5,
                            "pedestrian group": 1.1e6,
                            "two wheeler": 2.7e5,
                            "large vehicle": 9e5,
                            "background": 1.3e8}

        return radar_point_dict

    @staticmethod
    def get_class_weights():
        """ Calculates a weighting factor for each class based on the total number of points of each class in the RadarScenes dataset.
        """
        radar_point_dict = ClassDistribution.get_radar_point_dict()
        all = radar_point_dict.get("car") + radar_point_dict.get("pedestrian") + radar_point_dict.get("pedestrian group") + \
            radar_point_dict.get("two wheeler") + radar_point_dict.get("large vehicle") + radar_point_dict.get("background")

        weight_background = ((all / radar_point_dict.get("background"))) / (all / radar_point_dict.get("two wheeler"))
        weight_car = ((all / radar_point_dict.get("car"))) / (all / radar_point_dict.get("two wheeler"))
        weight_ped = ((all / radar_point_dict.get("pedestrian"))) / (all / radar_point_dict.get("two wheeler"))
        weight_ped_group = ((all / radar_point_dict.get("pedestrian group"))) / (all / radar_point_dict.get("two wheeler"))
        weight_larg_veh = ((all / radar_point_dict.get("large vehicle"))) / (all / radar_point_dict.get("two wheeler"))
        weight_two_wheeler = ((all / radar_point_dict.get("two wheeler"))) / (all / radar_point_dict.get("two wheeler"))

        class_weight_dict = {"car": weight_car,
                             "pedestrian": weight_ped,
                             "pedestrian group": weight_ped_group,
                             "two wheeler": weight_two_wheeler,
                             "large vehicle": weight_larg_veh,
                             "background": weight_background}

        return class_weight_dict
