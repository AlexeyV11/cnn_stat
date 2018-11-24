import math

class BarChart:
    char_tl = "╒"
    char_bl = "╘"
    char_tr = "╕"
    char_br = "╛"
    char_top_cross = "╤"  # cross
    char_cross = "┼"
    char_bottom_cross = "╧"
    char_horizontal_double = "═"
    char_horizontal_single = "─"
    char_vetrtical = "│"

    char_cross_left = "├"
    char_cross_right = "┤"

    def __init__(self, values, header=None, name=None, width_bar=80):
        self._values = values
        self._header = header
        self._name = name

        self._width_bar = width_bar

    def _add_separator_line(self, arr_width, first=False, last=False):
        # they can't be true together
        assert((first and last) == False)

        if first:
            b = BarChart.char_tl
            m = BarChart.char_top_cross
            e = BarChart.char_tr
            hor = BarChart.char_horizontal_double
        elif last:
            b = BarChart.char_bl
            m = BarChart.char_bottom_cross
            e = BarChart.char_br
            hor = BarChart.char_horizontal_double
        else:
            b = BarChart.char_cross_left
            m = BarChart.char_cross
            e = BarChart.char_cross_right
            hor = BarChart.char_horizontal_single

        line = b
        for i in range(len(arr_width)):
            line += hor * arr_width[i] + (m if i != len(arr_width) - 1 else e)

        return line + "\n"

    def plot(self):



        k = max(self._values)

        leng_header = max([len(s) for s in self._header])
        result = self._name  + "\n"

        result += self._add_separator_line([leng_header, self._width_bar], first=True)

        for i in range(len(self._header)):

            result += BarChart.char_vetrtical
            result += ("{:" + str(leng_header) + "}").format(self._header[i])
            result += BarChart.char_vetrtical
            result += ("{:" + str(self._width_bar) + "}").format("\u25AE" * math.ceil(self._values[i] / k * 80))
            result += BarChart.char_vetrtical
            result += "\n"
            result += self._add_separator_line([leng_header, self._width_bar], last=(i == len(self._header) - 1))

        return result
