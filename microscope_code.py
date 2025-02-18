# from labjack import ljm  # NOTE: MUST COMMENT BACK IN LAB

import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory

# from ttkthemes import ThemedTk  # NOTE FIGURE OUT

import time

# import serial
from datetime import date
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)  # add plot and toolbar to GUI

# Logger:
import logging

# Packages for ETA backend
import json

# import etabackend.eta  # Available at: https://github.com/timetag/ETA, https://eta.readthedocs.io/en/latest/

from pathlib import Path
import re

# --------
import time
import socket
import pickle
import copy
import threading

# from Swabian_Microscope_library_copy1028 import draw_image_heatmap_ratio


class GUI:

    def __init__(self):

        # Create and configure the main GUI window
        self.init_window()

        # define global variables
        self.init_parameters()
        self.x = None
        self.y = None
        self.z = None

        # Create and place tabs frame on window grid
        self.init_fill_tabs()
        self.live_mode = True  # FIXME: add button to change this

    def init_window(self):
        self.root = tk.Tk()
        self.root.title("Quantum Microscope GUI")  # *Ghostly matters*
        # self.root.resizable(True, True)
        # self.root.config(background='#0a50f5')   # TODO figure out why colors don't work
        self.root.geometry("1200x1200")

    def init_parameters(self):
        # TODO: CHECK WHAT WE CAN REMOVE!!!
        self.tabControl = None  # for plot we destoy occasionally
        self.data = []
        self.running = False  # this tracks if we are running a scan (collecting counts from detector)
        self.demo_connect = (
            False  # temp for demo to check if we've actually connected to device
        )
        self.current_file_name = None
        self.current_file_type = None
        self.current_file_path = None
        self.widgets = {}
        self.button_color = "grey"  # default button colors
        self.port = tk.StringVar()  # note maybe change later when implemented
        self.params = {
            "nr_pixels": {
                "var": tk.IntVar(value=4),
                "type": "int entry",
                "default": 4,
                "value": [1, 2, 4],
            },
            "file_name": {
                "var": tk.StringVar(value=""),
                "type": "str entry",
                "default": "ToF_Bio_cap_10MHz_det1_marker_ch4_10.0ms_[2.1, 2.45, -1.4, -3.4]_100x100_231218.timeres",
                "value": [
                    "ToF_terra_10MHz_det2_10.0ms_[2.1, 2.5, -3.2, -4.8]_100x100_231030.timeres",
                    "ToF_terra_10MHz_det2_1.0ms_[2.1, 3.9, -3.2, -4.8]_100x100_231102.timeres",
                    "ToF_terra_10MHz_det2_0.5ms_[2.1, 3.9, -3.2, -4.8]_100x100_231102.timeres",
                ],
            },
            # 'folder_name': {'var': tk.StringVar(),       'type': 'str entry', 'default': '',  'value': ['~/Desktop/GUI/Data1', '~/Desktop/GUI/Data2', '~/Desktop/GUI/Data3']},
            "eta_recipe": {
                "var": tk.StringVar(value=""),
                "type": "str entry",
                "default": "3D_tof_swabian_marker_ch4.eta",
                "value": [
                    "3D_tof_swabian_marker_ch4.eta",
                    "lifetime_new_spectrometer_4_ch_lifetime.eta",
                    "lifetime_det1_spectrometer_tof.eta",
                ],
            },
        }
        self.ch_bias_list = []
        self.ch_trig_list = []
        self.logger_box = None  # unsure if we should define it here at all

        # ANALYSIS PARAMS:
        self.eta_recipe = tk.StringVar(
            value="/Users/juliawollter/Desktop/Microscope GUI/Swabian_multiframe_recipe_bidirectional_segments_0.0.4.eta"
        )  # value='C:/Users/vLab/Desktop/Spectra GUI  Julia/LIDAR GUI/Recipes/3D_tof_swabian_marker_ch4.eta')
        self.data_folder = tk.StringVar(
            value="/Users/juliawollter/Desktop/Microscope GUI/Data"
        )
        self.clue = tk.StringVar(value="digit6")
        self.bins = tk.IntVar(value=20000)  # bins*binssize = 1/frep [ps]
        self.ch_sel = tk.StringVar(value="h2")
        self.save_folder = tk.StringVar(
            value="/Users/juliawollter/Desktop/Microscope GUI/Data/Analysis"
        )  # where images, gifs and analysis is saved

        # SCAN PARAMS:
        # self.dimX = tk.IntVar(value=100)
        self.dimY = tk.IntVar(value=100)
        self.freq = tk.DoubleVar(value=5.0)
        self.nr_frames = tk.IntVar(value=1)
        self.ampX = tk.DoubleVar(value=0.3)  # --> step values between -0.3 and 0.3
        self.ampY = tk.DoubleVar(value=0.3)  # --> sine values between -0.3 and 0.3
        self.data_folder = tk.StringVar(
            value="/Users/juliawollter/Desktop/Microscope GUI/Data"
        )
        self.data_file = tk.StringVar(
            value="digit6_sineFreq(1.0)_numFrames(10)_sineAmp(0.3)_stepAmp(0.3)_stepDim(100)_date(240114)_time(22h20m23s).timeres"
        )

        # -----------


class NBControl:
    @staticmethod
    def add_tab(parent_nb, tab_name, child_tab=None):
        if not child_tab:
            child_tab = ttk.Frame(parent_nb, borderwidth=1, relief=tk.FLAT)  # TODO
        parent_nb.add(child_tab, text=tab_name)
        return child_tab

    @staticmethod
    def add_notebook(parent_tab, **kwargs):
        notebook = ttk.Notebook(parent_tab)
        notebook.pack(**kwargs)
        return notebook


class ScanTab:
    # FOR EACH TAB WE WANT TO BUILD IT AND BE ABLE TO CREATE ZOOMED TABS
    def __init__(self, tab, tabname, new=True):
        self.tabname = tabname
        self.tab = self.root = tab
        self.nr_of_children = 0

        # tk.Button(tab, text="Destroy Tab", activeforeground='blue', background='red', command=self.destroy_tab).grid(row=0, column=0, sticky='w')  # FOR ADDING MORE TABS
        # ttk.Button(tab, text="Destroy Tab", command=self.destroy_tab).grid(row=2, column=0, rowspan=5, sticky='w')  # FOR ADDING MORE TABS

        # INITIALIZE GUI INSTANCE HERE
        self.init_parameters()

        self.init_fill_tabs()

        self.t7 = T7(self)

        # Below is just because we default to slow mode and want to show or convert the max and min directly
        self.minX.set(round(-self.ampX.get() + self.t7.x_offset, 10))
        self.maxX.set(round(self.ampX.get() + self.t7.x_offset, 10))
        self.minY.set(round(-self.ampY.get() + self.t7.y_offset, 10))
        self.maxY.set(round(self.ampY.get() + self.t7.y_offset, 10))

    def destroy_tab(self):
        self.root.destroy()

    def init_parameters(self):
        self.parent = None

        # TODO: CHECK WHAT WE CAN REMOVE!!!
        self.tabControl = None  # for plot we destoy occasionally
        self.data = []
        self.state = tk.StringVar(
            value="inactive"
        )  # this tracks if we are running a scan (collecting counts from detector)
        self.demo_connect = (
            False  # temp for demo to check if we've actually connected to device
        )
        self.current_file_name = None
        self.current_file_type = None
        self.current_file_path = None
        self.widgets = {}
        self.button_color = "white"  # default button colors
        self.port = tk.StringVar()  # note maybe change later when implemented
        self.params = {
            "nr_pixels": {
                "var": tk.IntVar(value=4),
                "type": "int entry",
                "default": 4,
                "value": [1, 2, 4],
            },
            "file_name": {
                "var": tk.StringVar(value=""),
                "type": "str entry",
                "default": "ToF_Bio_cap_10MHz_det1_marker_ch4_10.0ms_[2.1, 2.45, -1.4, -3.4]_100x100_231218.timeres",
                "value": [
                    "ToF_terra_10MHz_det2_10.0ms_[2.1, 2.5, -3.2, -4.8]_100x100_231030.timeres",
                    "ToF_terra_10MHz_det2_1.0ms_[2.1, 3.9, -3.2, -4.8]_100x100_231102.timeres",
                    "ToF_terra_10MHz_det2_0.5ms_[2.1, 3.9, -3.2, -4.8]_100x100_231102.timeres",
                ],
            },
            # 'folder_name': {'var': tk.StringVar(),       'type': 'str entry', 'default': '',  'value': ['~/Desktop/GUI/Data1', '~/Desktop/GUI/Data2', '~/Desktop/GUI/Data3']},
            "eta_recipe": {
                "var": tk.StringVar(value=""),
                "type": "str entry",
                "default": "3D_tof_swabian_marker_ch4.eta",
                "value": [
                    "3D_tof_swabian_marker_ch4.eta",
                    "lifetime_new_spectrometer_4_ch_lifetime.eta",
                    "lifetime_det1_spectrometer_tof.eta",
                ],
            },
        }
        self.ch_bias_list = []
        self.ch_trig_list = []
        self.logger_box = None  # unsure if we should define it here at all

        self.x_move_var = tk.IntVar(value=1)  # which x pixel we move to
        self.y_move_var = tk.IntVar(value=1)
        # ANALYSIS PARAMS:
        # self.eta_recipe = tk.StringVar(value='Swabian_multiframe_recipe_bidirectional_segments_marker4_20.eta')  # value='C:/Users/vLab/Desktop/Spectra GUI  Julia/LIDAR GUI/Recipes/3D_tof_swabian_marker_ch4.eta')
        self.clue = tk.StringVar(value="test")
        self.bins = tk.IntVar(value=20000)  # bins*binssize = 1/frep [ps]
        self.ch_sel = tk.StringVar(value="h2")
        self.eta_recipe = tk.StringVar(
            value="Swabian_multiframe_recipe_bidirectional_segments_marker4_28.eta"
        )  # value='C:/Users/vLab/Desktop/Spectra GUI  Julia/LIDAR GUI/Recipes/3D_tof_swabian_marker_ch4.eta')
        self.eta_recipe_slow = "Swabian_slow_multiframe_recipe_bidirectional_segments_marker4_28.eta"  # value='C:/Users/vLab/Desktop/Spectra GUI  Julia/LIDAR GUI/Recipes/3D_tof_swabian_marker_ch4.eta')
        self.eta_recipe_fast = "Swabian_multiframe_recipe_bidirectional_segments_marker4_28.eta"  # value='C:/Users/vLab/Desktop/Spectra GUI  Julia/LIDAR GUI/Recipes/3D_tof_swabian_marker_ch4.eta')
        # self.anal_data_file = tk.StringVar(value='/Users/juliawollter/Desktop/GUI Micro/Data/240805/scan1_scantime(500.0)_intTime(0.01)_sineAmp(0.2)_stepAmp(0.2)_stepDim(100)_date(240805)_time(16h48m58s).timeres')
        self.anal_data_file = tk.StringVar(
            value="Data/240918/hBN1000_scantime(100.0)_dwellTime(0.01)_sineAmp(0.2)_stepAmp(0.2)_stepDim(100)_date(240918)_time(15h14m56s).timeres"
        )

        self.save_folder = tk.StringVar(
            value="/Analysis"
        )  # where images, gifs and analysis is saved

        # SCAN PARAMS:
        # self.dimX = tk.IntVar(value=100)
        self.sweep_mode = tk.StringVar(value="linear")
        self.dimY = tk.IntVar(value=100)
        self.dimX = self.dimY  # NOTE: WILL IMPLEMENT IT's OWN VARIABLE LATER

        self.int_time = tk.DoubleVar(value=0.005)
        self.freq = tk.DoubleVar(value=1.0)
        self.nr_frames = tk.IntVar(value=1)
        self.ampX = tk.DoubleVar(value=0.3)  # --> step values between -0.3 and 0.3
        self.ampY = tk.DoubleVar(value=0.3)  # --> sine values between -0.3 and 0.3

        self.minX = tk.DoubleVar(value=0)
        self.maxX = tk.DoubleVar(value=0)
        self.minY = tk.DoubleVar(value=0)
        self.maxY = tk.DoubleVar(value=0)

        self.scopelength = tk.DoubleVar(value=100)
        self.lensslope = tk.DoubleVar(value=313)
        self.scopelength.set(round(self.ampX.get() * self.lensslope.get(), 10))
        self.stepsize = tk.DoubleVar(value=30)
        self.stepsize.set(round(self.scopelength.get() * 1000 / self.dimY.get(), 5))

        self.heatmapratio = tk.DoubleVar(value=0.5)

        self.data_folder = tk.StringVar(
            value=f'K:/Microscope/Data/{date.today().strftime("%y%m%d")}'
        )
        self.data_file = tk.StringVar(value="")
        self.speed_mode = tk.StringVar(value="slow")

        # -----------

    def init_fill_tabs(self, tabname="New Scan"):

        # TABS STYLE
        if False:
            style1 = ttk.Style()
            style1.theme_create(
                "style1",
                parent="alt",
                settings={
                    "TNotebook": {"configure": {"tabmargins": [0, 0, 0, 0]}},
                    "TNotebook.Tab": {
                        "configure": {
                            "padding": [10, 10],
                            "font": ("garamond", "11", "bold"),
                        },
                    },
                },
            )
            style1.theme_use("style1")
            TROUGH_COLOR = "white"
            BAR_COLOR = "blue"
            style1.configure(
                "bar.Horizontal.TProgressbar",
                troughcolor=TROUGH_COLOR,
                bordercolor=TROUGH_COLOR,
                background=BAR_COLOR,
                lightcolor=BAR_COLOR,
                darkcolor=BAR_COLOR,
            )
            # style1.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
            # style1.configure("red.Horizontal.TProgressbar", troughcolor='gray', background='red')  # progressbar!!

        # ----Create notebooks for multi tab window:----

        # ADD TABS TO MANAGER 2
        # tabControl2 = ttk.Notebook(parent)

        # LIVE SCAN TAB: (but not live yet!!!!)
        live_tab = ttk.Frame(self.tab)

        live_tab_1 = ttk.Frame(live_tab)
        live_tab_2 = ttk.Frame(live_tab)
        live_tab_3 = ttk.Frame(live_tab)

        ttk.Label(live_tab_1, text="Scan Settings", font=("", 15), width=20).grid(
            row=0, column=0, columnspan=2, sticky="news", padx=0, pady=0
        )

        #   Data acquisition (pre scan)
        scan_conf = self.choose_scan_configs_widget(live_tab_1)
        scan_conf.grid(row=1, column=0, columnspan=2, sticky="news", padx=2, pady=1)

        # Data processing (post scan)
        anal_conf = self.choose_analysis_configs_widget(live_tab_1)
        anal_conf.grid(
            row=2, column=0, columnspan=2, sticky="news", padx=2, pady=1
        )  # in sub frame

        # Start scan button
        self.start_scan_widget(
            anal_conf, scan_conf
        )  # .grid(row=3, column=0, columnspan=2, sticky="news", padx=2, pady=0)  # in sub frame

        # PLOTS AND MOVE
        self.plot_analysis_widget(live_tab_2).grid(
            row=0, column=2, sticky="news", padx=2, pady=0
        )  # in sub frame

        # self.move_with_plot_widget(live_tab_2).grid(row=1, column=2, sticky="new", padx=2, pady=0)  # in sub frame
        self.move_with_plot_widget(live_tab_1).grid(
            row=10, column=0, columnspan=2, sticky="news", padx=2, pady=0
        )  # in sub frame
        self.set_scope_length_widget(live_tab_1).grid(
            row=11, column=0, columnspan=2, sticky="news", padx=2, pady=0
        )
        self.set_stepsize_widget(live_tab_1).grid(
            row=12, column=0, columnspan=2, sticky="news", padx=2, pady=0
        )
        self.heatmap_ratio_widget(live_tab_1).grid(
            row=13, column=0, columnspan=2, sticky="news", padx=2, pady=0
        )
        # Logged information:
        self.log_scan_widget(live_tab_3).grid(
            row=4, column=0, columnspan=2, rowspan=20, sticky="news", padx=2, pady=0
        )  # in sub frame

        live_tab_1.grid(row=0, column=1, sticky="news", padx=1)
        live_tab_2.grid(row=0, column=2, sticky="news", padx=1, rowspan=3)
        live_tab_3.grid(row=0, column=0, sticky="n", padx=1)

        ttk.Button(live_tab, text="Destroy Tab", command=self.destroy_tab).grid(
            row=2, column=0, sticky="w"
        )  # FOR ADDING MORE TABS

        # self.tabControl.add(live_tab, text=tabname)
        live_tab.grid(row=1, column=0, sticky="w")
        # tabControl2.grid(row=0, column=0, sticky='nesw', pady=5)

    # ---------------

    @staticmethod
    def add_to_grid(widg, rows, cols, sticky, columnspan=None):
        for i in range(len(widg)):
            if columnspan:
                widg[i].grid(
                    row=rows[i],
                    column=cols[i],
                    sticky=sticky[i],
                    padx=0.1,
                    pady=0.1,
                    columnspan=columnspan[i],
                )
            else:
                widg[i].grid(
                    row=rows[i], column=cols[i], sticky=sticky[i], padx=0.1, pady=0.1
                )

    def select_speed(self):
        if self.speed_mode.get() == "fast":
            self.eta_recipe.set(self.eta_recipe_fast)
            self.variable_containers["int time"]["lab"].configure(text="freq (Hz)")
            self.variable_containers["int time"]["entry"].configure(
                textvariable=self.freq
            )
            self.variable_containers["min X"]["entry"].configure(state="disabled")
            self.variable_containers["max X"]["entry"].configure(state="disabled")
            self.variable_containers["min Y"]["entry"].configure(state="disabled")
            self.variable_containers["max Y"]["entry"].configure(state="disabled")

            self.variable_containers["amp X"]["entry"].configure(state="normal")
            self.variable_containers["amp Y"]["entry"].configure(state="normal")

        elif self.speed_mode.get() == "slow":
            self.eta_recipe.set(self.eta_recipe_slow)
            self.variable_containers["int time"]["lab"].configure(text="dwell (s)")
            self.variable_containers["int time"]["entry"].configure(
                textvariable=self.int_time
            )

            self.variable_containers["min X"]["entry"].configure(state="disabled")
            self.variable_containers["max X"]["entry"].configure(state="disabled")
            self.variable_containers["min Y"]["entry"].configure(state="disabled")
            self.variable_containers["max Y"]["entry"].configure(state="disabled")

            self.variable_containers["amp X"]["entry"].configure(state="normal")
            self.variable_containers["amp Y"]["entry"].configure(state="normal")

        elif self.speed_mode.get() == "zoom":
            self.eta_recipe.set(self.eta_recipe_slow)
            self.variable_containers["int time"]["lab"].configure(text="dwell (s)")
            self.variable_containers["int time"]["entry"].configure(
                textvariable=self.int_time
            )

            self.variable_containers["min X"]["entry"].configure(state="normal")
            self.variable_containers["max X"]["entry"].configure(state="normal")
            self.variable_containers["min Y"]["entry"].configure(state="normal")
            self.variable_containers["max Y"]["entry"].configure(state="normal")

            self.variable_containers["amp X"]["entry"].configure(state="disabled")
            self.variable_containers["amp Y"]["entry"].configure(state="disabled")

            # change x_amp and y_amp
            # add x min, max and y min, max

        # self.logger_box.module_logger.info(f"Chosen speed: {self.speed_mode.get()}")
        self.suggest_name()

    def suggest_name(self):
        if (
            self.speed_mode.get() == "slow"
        ):  # note: removed nr of frame from calculation for scantime
            scantime = round(self.dimY.get() * self.dimX.get() * self.int_time.get(), 1)
            filename = (
                f"{self.clue.get()}_"
                f'date({date.today().strftime("%y%m%d")})_'
                f'time({time.strftime("%Hh%Mm%Ss", time.localtime())})_'
                f"scantime({scantime})_"
                f"dwellTime({self.int_time.get()})_"
                f"xAmp({self.ampX.get()})_"
                f"yAmp({self.ampY.get()})_"
                f"xyDim({self.dimY.get()}).timeres"
            )
        elif (
            self.speed_mode.get() == "fast"
        ):  # note: removed nr of frame from calculation for scantime
            scantime = round(self.dimY.get() / self.freq.get(), 2)
            filename = (
                f"{self.clue.get()}_"
                f"scantime({scantime})_"
                f"sineFreq({self.freq.get()})_"
                f"xAmp({self.ampX.get()})_"
                f"yAmp({self.ampY.get()})_"
                f"xyDim({self.dimY.get()})_"
                f'date({date.today().strftime("%y%m%d")})_'
                f'time({time.strftime("%Hh%Mm%Ss", time.localtime())}).timeres'
            )
        elif (
            self.speed_mode.get() == "zoom"
        ):  # note: removed nr of frame from calculation for scantime
            scantime = round(self.dimY.get() * self.dimX.get() * self.int_time.get(), 1)
            filename = (
                f"{self.clue.get()}_"
                f'date({date.today().strftime("%y%m%d")})_'
                f'time({time.strftime("%Hh%Mm%Ss", time.localtime())})_'
                f"scantime({scantime})_"
                f"dwellTime({self.int_time.get()})_"
                f"xMin({self.minX.get()})_"
                f"xMax({self.maxX.get()})_"
                f"yMin({self.minY.get()})_"
                f"yMax({self.maxY.get()})_"
                f"xyDim({self.dimY.get()}).timeres"
            )

        self.variable_containers["scantime title"]["lab"].config(
            text=f"Scan time = {scantime} sec ({round(scantime / 60, 1)} min)"
        )
        self.variable_containers["filename"]["entry"].delete(0, tk.END)
        self.variable_containers["filename"]["entry"].insert(0, filename)

        self.data_file.set(filename)

    def choose_scan_configs_widget(self, tab):

        def open_folder():
            file = askdirectory()
            variables["data folder"]["entry"].delete(0, tk.END)
            variables["data folder"]["entry"].insert(0, file)

        """def init_plot(show, figsize=(2, 2)):
            ##---
            x_l = []
            y_l = []
            fig, ax = plt.subplots(1, figsize=figsize, frameon=False)
            #ax.set_frame_on(False)
            if show:
                ax.plot(x_l, y_l, 'b-')
                ax.set_axis_off()
                fig.canvas.draw_idle()  # updates the canvas immediately?
            return fig, ax

        # TODO: ADD SLOW MODE
        def plot_path():
            if self.speed_mode == 'fast':
                step_vals = []  # y-vals

                step_size = (2 * variables["amp Y"]["var"].get()) / (
                            variables["dim Y"]["var"].get() - 1)  # step size of our x values
                k = -1 * variables["amp Y"]["var"].get()
                for i in range(variables["dim Y"]["var"].get()):
                    step_vals.append(round(k + self.t7.y_offset, 10))
                    k += step_size

                x_min = - variables["amp X"]["var"].get() / 2
                x_max = variables["amp X"]["var"].get() / 2

                x_l = []
                y_l = []
                for j in range(0, len(step_vals), 2):
                    x_l += [x_min, x_max, x_max, x_min]
                    y_l += [step_vals[j], step_vals[j], step_vals[j + 1], step_vals[j + 1]]

                ax1.clear()
                ax1.set_axis_off()
                ax1.plot(x_l, y_l, 'b-')
                fig1.canvas.draw_idle()  # updates the canvas immediately?
                self.logger_box.module_logger.info("Done plotting")
            else:
                print("TODO IMPLEMENT SLOW MODE PLOT")"""

        def select_demo():
            if self.demo_mode.get() is True:
                self.diagnostics_mode.set(False)
                self.record_mode.set(False)
            else:
                self.record_mode.set(True)

        def select_record():
            if self.record_mode.get() is True:
                self.demo_mode.set(False)
            else:
                self.demo_mode.set(True)
                self.diagnostics_mode.set(False)

        def select_diagnostics():
            if self.diagnostics_mode.get() is True:
                self.record_mode.set(True)  # for diagnostics, we must be in record mode
            self.demo_mode.set(False)

        def changed_params(event, k):
            # print("Changed params", event.widget.get())
            self.suggest_name()
            print("key =", k)
            if k in ["amp X", "amp Y", "min X", "min Y"]:
                update_scan_range(k, event.widget.get())

        def update_scan_range(key, val):
            print("UPDATE SCAN RANGE OF", key, "WITH", val)
            if key == "amp X":
                self.minX.set(round(-eval(val) + self.t7.x_offset, 10))
                self.maxX.set(round(eval(val) + self.t7.x_offset, 10))
                self.scopelength.set(self.ampX.get() * self.lensslope.get())
                print(self.minX.get())
                print(self.maxX.get())
            elif key == "amp Y":
                self.minY.set(round(-eval(val) + self.t7.y_offset, 10))
                self.maxY.set(round(eval(val) + self.t7.y_offset, 10))
                print(self.minY.get())
                print(self.maxY.get())

            elif key == "min X" or key == "max X":
                print(self.t7.prev_speed_mode)
                self.ampX.set(round((self.maxX.get() - self.minX.get()) / 2, 10))

            elif key == "min Y" or key == "max Y":
                print(self.t7.prev_speed_mode)
                self.ampY.set(round((self.maxY.get() - self.minY.get()) / 2, 10))
            else:
                print("KEY", key, "NOT RECOGNIZED")

        def validate_all_float(
            action,
            index,
            value_if_allowed,
            prior_value,
            text,
            validation_type,
            trigger_type,
            widget_name,
        ):
            # SOURCE: https://stackoverflow.com/questions/8959815/restricting-the-value-in-tkinter-entry-widget
            if action == "1":  # # action=1 -> insert
                # print("INSERT OBSERVED:", text)
                if text in "0123456789.-":
                    try:
                        float(value_if_allowed)
                        return True
                    except ValueError:
                        print("(all float) failed", value_if_allowed)
                        return False
                else:
                    return False
            else:
                return True

        def validate_pos_float(
            action,
            index,
            value_if_allowed,
            prior_value,
            text,
            validation_type,
            trigger_type,
            widget_name,
        ):
            # SOURCE: https://stackoverflow.com/questions/8959815/restricting-the-value-in-tkinter-entry-widget
            if action == "1":  # # action=1 -> insert
                # print("INSERT OBSERVED:", text)
                if text in "0123456789.":
                    try:
                        float(value_if_allowed)
                        return True
                    except ValueError:
                        return False
                else:
                    return False
            else:
                print("(pos float)", value_if_allowed)
                return True

        def validate_pos_int(
            action,
            index,
            value_if_allowed,
            prior_value,
            text,
            validation_type,
            trigger_type,
            widget_name,
        ):
            # SOURCE: https://stackoverflow.com/questions/8959815/restricting-the-value-in-tkinter-entry-widget
            if action == "1":  # # action=1 -> insert
                # print("INSERT OBSERVED:", text)
                if text in "0123456789":
                    try:
                        int(value_if_allowed)
                        return True
                    except ValueError:
                        return False
                else:
                    return False
            else:
                return True

        frm_misc = ttk.Frame(tab, relief=tk.RAISED)

        # ---- PLOT PATH: ----
        # frm_path = ttk.Frame(tab, relief=tk.RAISED)
        # fig1, ax1 = init_plot(show=True, figsize=(1, 1))
        # plt_frame1, canvas1 = self.pack_plot(frm_path, fig1, toolbar=True)  # plt_frame1.grid(row=1, column=3, rowspan=4, columnspan=4, sticky="news", padx=0, pady=0)

        # Button to plot path
        # path_butt = ttk.Button(frm_path, text="plot path", command=plot_path, activeforeground='blue',  highlightbackground=self.button_color)

        # plt_frame1.grid(row=1, column=0, columnspan=10, sticky="e", padx=0, pady=0)
        # path_butt.grid(row=0, column=0, sticky='ew', padx=0, pady=0)
        # frm_path.grid(row=50, column=0, columnspan=10, sticky='ew', padx=0, pady=0)
        # --------------------

        ttk.Label(frm_misc, text="Acquisition", font=("", 15)).grid(
            row=0, column=0, sticky="ew", padx=1, pady=0
        )

        theo_scan_lbl = ttk.Label(
            frm_misc, text=f"", font=("", 12)
        )  # {self.nr_frames.get() * self.dimY.get() * ((1/self.freq.get()))}  # {self.num_frames * self.step_dim * self.t7.step_delay}
        theo_scan_lbl.grid(row=11, column=1, columnspan=5, sticky="ew", padx=0, pady=0)

        vcmd_pos_flo = (
            tab.register(validate_pos_float),
            "%d",
            "%i",
            "%P",
            "%s",
            "%S",
            "%v",
            "%V",
            "%W",
        )  # for float validate function
        vcmd_all_flo = (
            tab.register(validate_all_float),
            "%d",
            "%i",
            "%P",
            "%s",
            "%S",
            "%v",
            "%V",
            "%W",
        )  # for float validate function including NEGATIVES
        vcmd_pos_int = (
            tab.register(validate_pos_int),
            "%d",
            "%i",
            "%P",
            "%s",
            "%S",
            "%v",
            "%V",
            "%W",
        )  # for float validate function including NEGATIVES

        variables = self.variable_containers = {
            "info": {
                "lab": ttk.Label(frm_misc, text="info"),
                "entry": ttk.Entry(frm_misc, textvariable=self.clue, width=50),
                "var": self.clue,
            },
            "scantime title": {"lab": theo_scan_lbl},
            "int time": {
                "lab": ttk.Label(frm_misc, text="dwell (s)"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.int_time,
                    width=5,
                    validate="key",
                    validatecommand=vcmd_pos_flo,
                ),
                "var": self.int_time,
            },
            "frequency": {
                "lab": ttk.Label(frm_misc, text="freq"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.freq,
                    width=5,
                    validate="key",
                    validatecommand=vcmd_pos_flo,
                ),
                "var": self.freq,
            },
            # (SET TO 1 FOR NOW)
            #'nr frames': {
            #    'entry': ttk.Entry(frm_misc, textvariable=self.nr_frames, width=15),
            #    'var': self.nr_frames},
            # (MIGHT ADD LATER) 'dim X':  {'entry': ttk.Entry(frm_misc, textvariable=self.dimX, width=15), 'var': self.dimX},
            "dim Y": {
                "lab": ttk.Label(frm_misc, text="dim Y"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.dimY,
                    width=5,
                    validate="key",
                    validatecommand=vcmd_pos_int,
                ),
                "var": self.dimY,
            },
            "amp X": {
                "lab": ttk.Label(frm_misc, text="amp X"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.ampX,
                    width=8,
                    validate="key",
                    validatecommand=vcmd_pos_flo,
                ),  # disabledforeground='red', disabledbackground='lightgray'),
                "var": self.ampX,
            },
            "amp Y": {
                "lab": ttk.Label(frm_misc, text="amp Y"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.ampY,
                    width=8,
                    validate="key",
                    validatecommand=vcmd_pos_flo,
                ),  # disabledforeground='red', disabledbackground='lightgray'),
                "var": self.ampY,
            },
            "min X": {
                "lab": ttk.Label(frm_misc, text="min X"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.minX,
                    width=5,
                    state="disabled",
                    validate="key",
                    validatecommand=vcmd_all_flo,
                ),  # disabledforeground='red', disabledbackground='lightgray'),
                "var": self.minX,
            },
            "max X": {
                "lab": ttk.Label(frm_misc, text="max X"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.maxX,
                    width=5,
                    state="disabled",
                    validate="key",
                    validatecommand=vcmd_all_flo,
                ),  # disabledforeground='red', disabledbackground='lightgray'),
                "var": self.maxX,
            },
            "min Y": {
                "lab": ttk.Label(frm_misc, text="min Y"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.minY,
                    width=5,
                    state="disabled",
                    validate="key",
                    validatecommand=vcmd_all_flo,
                ),  # disabledforeground='red', disabledbackground='lightgray'),
                "var": self.minY,
            },
            "max Y": {
                "lab": ttk.Label(frm_misc, text="max Y"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.maxY,
                    width=5,
                    state="disabled",
                    validate="key",
                    validatecommand=vcmd_all_flo,
                ),  # disabledforeground='red', disabledbackground='lightgray'),
                "var": self.maxY,
            },
            "filename": {
                "butt": ttk.Button(
                    frm_misc, text="filename", command=self.suggest_name
                ),  # , activeforeground='blue', highlightbackground=self.button_color),
                "entry": ttk.Entry(
                    frm_misc, textvariable=self.data_file, width=10, state="readonly"
                ),  # foreground='blue'),  # , readonlybackground='lightgray'),  # foreground=text
                "var": self.data_file,
            },
            "data folder": {
                "butt": ttk.Button(
                    frm_misc, text="data folder", command=open_folder
                ),  # , activeforeground='blue', highlightbackground=self.button_color),
                "entry": ttk.Entry(frm_misc, textvariable=self.data_folder, width=15),
                "var": self.data_folder,
            },
            "lensslope": {
                "lab": ttk.Label(frm_misc, text="lensslope"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.lensslope,
                    width=8,
                    validate="key",
                    validatecommand=vcmd_pos_flo,
                ),  # disabledforeground='red', disabledbackground='lightgray'),
                "var": self.lensslope,
            },
            "scopelength": {
                "lab": ttk.Label(frm_misc, text="scopelength"),
                "entry": ttk.Entry(
                    frm_misc,
                    textvariable=self.scopelength,
                    width=8,
                    validate="key",
                    validatecommand=vcmd_pos_flo,
                ),  # disabledforeground='red', disabledbackground='lightgray'),
                "var": self.scopelength,
            },
        }

        # KEY BINDINGS FOR INPUT PARAMS:
        for key in variables.keys():
            if "entry" in variables[key].keys() and key not in [
                "filename",
                "data folder",
            ]:
                variables[key]["entry"].bind(
                    "<KeyRelease>", lambda event, k=key: changed_params(event, k)
                )

        self.slow_radiobutt = ttk.Radiobutton(
            frm_misc,
            text="slow",
            value="slow",
            variable=self.speed_mode,
            command=self.select_speed,
        )

        # ttk.Radiobutton(frm_misc, text="fast", value="fast", variable=self.speed_mode, command=self.select_speed, state='disabled').grid(row=1, column=1)
        # ^TODO: reenable fast mode when it's fixed and improved

        self.zoom_radiobutt = ttk.Radiobutton(
            frm_misc,
            text="zoom",
            value="zoom",
            variable=self.speed_mode,
            command=self.select_speed,
            state="disabled",
        )

        self.slow_radiobutt.grid(row=1, column=0)
        self.zoom_radiobutt.grid(row=1, column=2)

        i = 2

        variables["info"]["lab"].grid(row=i, column=0, sticky="e", padx=0, pady=0)
        variables["info"]["entry"].grid(
            row=i, column=1, columnspan=8, sticky="w", padx=1, pady=0
        )

        variables["int time"]["lab"].grid(
            row=i + 1, column=0, sticky="e", padx=0, pady=0
        )
        variables["int time"]["entry"].grid(
            row=i + 1, column=1, sticky="w", padx=0, pady=0
        )

        variables["dim Y"]["lab"].grid(row=i + 2, column=0, sticky="e", padx=0, pady=0)
        variables["dim Y"]["entry"].grid(
            row=i + 2, column=1, sticky="w", padx=0, pady=0
        )

        variables["amp X"]["lab"].grid(row=i + 1, column=2, sticky="e", padx=0, pady=0)
        variables["amp X"]["entry"].grid(
            row=i + 1, column=3, sticky="w", padx=0, pady=0
        )
        variables["amp Y"]["lab"].grid(row=i + 2, column=2, sticky="e", padx=0, pady=0)
        variables["amp Y"]["entry"].grid(
            row=i + 2, column=3, sticky="w", padx=0, pady=0
        )

        variables["min X"]["lab"].grid(row=i + 1, column=4, sticky="e", padx=0, pady=0)
        variables["min X"]["entry"].grid(
            row=i + 1, column=5, sticky="w", padx=0, pady=0
        )
        variables["min Y"]["lab"].grid(row=i + 2, column=4, sticky="e", padx=0, pady=0)
        variables["min Y"]["entry"].grid(
            row=i + 2, column=5, sticky="w", padx=0, pady=0
        )

        variables["max X"]["lab"].grid(row=i + 1, column=6, sticky="e", padx=0, pady=0)
        variables["max X"]["entry"].grid(
            row=i + 1, column=7, sticky="w", padx=0, pady=0
        )
        variables["max Y"]["lab"].grid(row=i + 2, column=6, sticky="e", padx=0, pady=0)
        variables["max Y"]["entry"].grid(
            row=i + 2, column=7, sticky="w", padx=0, pady=0
        )

        variables["filename"]["butt"].grid(
            row=i + 3, column=0, sticky="ew", padx=1, pady=0
        )
        variables["filename"]["entry"].grid(
            row=i + 3, column=1, columnspan=10, sticky="ew", padx=1, pady=0
        )

        variables["data folder"]["butt"].grid(
            row=i + 4, column=0, sticky="ew", padx=1, pady=0
        )
        variables["data folder"]["entry"].grid(
            row=i + 4, column=1, columnspan=10, sticky="ew", padx=1, pady=0
        )

        # ttk.Button(frm_misc, text="save to", command=open_folder, activeforeground='blue',
        #          highlightbackground=self.button_color).grid(row=8, column=0, sticky='ew', padx=0, pady=0)

        self.record_mode = tk.BooleanVar(value=False)
        self.diagnostics_mode = tk.BooleanVar(value=False)
        self.demo_mode = tk.BooleanVar(value=True)

        # tk.Checkbutton -->  anchor="w",
        variables["run scan check"] = {
            "butt": ttk.Checkbutton(
                frm_misc,
                text="Run Scan",
                command=select_record,
                variable=self.record_mode,
                onvalue=True,
                offvalue=False,
            )
        }
        variables["diagnostics check"] = {
            "butt": ttk.Checkbutton(
                frm_misc,
                text="Diagnostics (no data file)",
                command=select_diagnostics,
                variable=self.diagnostics_mode,
                onvalue=True,
                offvalue=False,
            )
        }
        variables["offline check"] = {
            "butt": ttk.Checkbutton(
                frm_misc,
                text="Offline",
                command=select_demo,
                variable=self.demo_mode,
                onvalue=True,
                offvalue=False,
            )
        }

        variables["run scan check"]["butt"].grid(
            row=10, column=0, sticky="ew", padx=1, pady=0
        )
        variables["diagnostics check"]["butt"].grid(
            row=10, column=1, columnspan=5, sticky="ew", padx=0, pady=0
        )
        variables["offline check"]["butt"].grid(
            row=10, column=6, columnspan=2, sticky="ew", padx=0.5, pady=0
        )

        self.suggest_name()

        return frm_misc

    def choose_analysis_configs_widget(self, tab):

        def get_recipe():
            recipe = askopenfilename(filetypes=[("ETA recipe", "*.eta")])
            self.eta_recipe.set(recipe)
            variables["eta_recipe"]["entry"].delete(0, tk.END)
            variables["eta_recipe"]["entry"].insert(0, recipe)

        def open_datafile():
            file = askopenfilename(filetypes=[("Timeres", "*.timeres")])
            self.anal_data_file.set(file)
            variables["file_name"]["entry"].delete(0, tk.END)
            variables["file_name"]["entry"].insert(0, file)

        def open_folder():
            file = askdirectory()
            variables["save_folder"]["entry"].delete(0, tk.END)
            variables["save_folder"]["entry"].insert(0, file)

        # def press_start_analysis():
        #    self.pb['value'] = 0
        #    self.root.update()  # testing

        frm_misc = ttk.Frame(tab, relief=tk.RAISED)
        ttk.Label(frm_misc, text="Analysis", font=("", 15)).grid(
            row=0, column=0, sticky="news", padx=1, pady=0
        )

        file_lab_parts = []
        file_entry = []

        variables = {
            "file_name": {
                "butt": ttk.Button(frm_misc, text="datafile", command=open_datafile),
                "entry": ttk.Entry(
                    frm_misc, textvariable=self.anal_data_file, width=50
                ),
                "var": self.save_folder,
            },
            "eta_recipe": {
                "butt": ttk.Button(frm_misc, text="eta recipe", command=get_recipe),
                "entry": ttk.Entry(frm_misc, textvariable=self.eta_recipe, width=50),
                "var": self.eta_recipe,
            },
            "save_folder": {
                "butt": ttk.Button(frm_misc, text="save folder", command=open_folder),
                "entry": ttk.Entry(frm_misc, textvariable=self.save_folder, width=50),
                "var": self.save_folder,
            },
            "bins": {
                "lab": ttk.Label(frm_misc, text="bins"),
                "entry": ttk.Entry(frm_misc, textvariable=self.bins, width=15),
                "var": self.bins,
            },
            "ch_sel": {
                "lab": ttk.Label(frm_misc, text="ch sel"),
                "entry": ttk.Entry(frm_misc, textvariable=self.ch_sel, width=15),
                "var": self.ch_sel,
            },
        }

        for i, key in enumerate(variables.keys()):
            # file_lab_parts.append(ttk.Label(frm_misc, text=label))
            # file_entry.append(variables[label]['entry'])
            # self.add_to_grid(widg=[file_lab_parts[i]], rows=[i + 1], cols=[0], sticky=["ew"])
            if key not in ["ch_sel", "bins"]:  # skip these for now
                self.add_to_grid(
                    widg=[variables[key]["entry"]],
                    rows=[i + 1],
                    cols=[1],
                    sticky=["ew"],
                )
                if "lab" in variables[key].keys():
                    self.add_to_grid(
                        widg=[variables[key]["lab"]],
                        rows=[i + 1],
                        cols=[0],
                        sticky=["ew"],
                    )
                elif "butt" in variables[key].keys():
                    self.add_to_grid(
                        widg=[variables[key]["butt"]],
                        rows=[i + 1],
                        cols=[0],
                        sticky=["ew"],
                    )

            if (
                key in self.variable_containers.keys()
            ):  # Adding local dict items to self.variable_containers dict
                print("DUPLICATE DICT KEY DETECTED!!!")
            else:
                self.variable_containers[key] = variables[key]

        # file_buts = [ttk.Button(frm_misc, text="datafile", command=open_datafile),  # ,  activeforeground='blue', highlightbackground=self.button_color),
        #             ttk.Button(frm_misc, text="eta recipe", command=get_recipe),  # ,   activeforeground='blue', highlightbackground=self.button_color),
        #             ttk.Button(frm_misc, text="save folder", command=open_folder),  # , activeforeground='blue', highlightbackground=self.button_color)
        #             ]
        # self.add_to_grid(widg=file_buts, rows=[1, 2, 3], cols=[0, 0, 0], sticky=["ew", "ew", "ew"])

        return frm_misc

    def update_progressbar(self, n=10):
        print("NOTE: NOT USING PROGRESSBAR ATM")
        return
        if self.pb["value"] < 100:
            self.pb["value"] = n + 1
            self.root.update()  # testing

    def scanning(self):  # TODO DO SOMETHING MORE HERE

        if self.state.get() == "run":  # if start button is active
            # self.get_counts()  # saves data to self.data. note that live graph updates every second using self.data
            # self.save_data(mode="a")
            self.logger_box.module_logger.info("running..")

            # pass
        elif self.state.get() == "cancel":
            self.logger_box.module_logger.info("cancelled..")
            self.state.set("inactive")

        else:
            # print("state =", self.state.get())
            pass

        # print("hi")
        self.root.after(1000, self.scanning)  # After 1 second, call scanning

    '''def save_data(self, data, folder="", filename='scan_matrix', mode='w'):
        """data_str = []
        for row in self.data:
            vals = [str(int(x)) for x in row]
            data_str.append(' '.join(vals) + ' \n')
        with open("counts_file.txt", mode) as file:  # FIXME need to make sure that new scan => new/empty file
            file.writelines(data_str)  # TODO maybe add time of each
        self.data = []  # removing data that is now saved in file"""
        if not os.path.exists(folder):  # If folder does not exist, create it
            os.makedirs(folder)
            print("FOLDER CREATED")

        data_str = []
        for row in data:
            vals = [str(int(x)) for x in row]
            data_str.append(' '.join(vals) + ' \n')
        with open(f"{folder}/data_{filename}.txt", mode) as file:  # FIXME need to make sure that new scan => new/empty file
            file.writelines(data_str)  # TODO maybe add time of each
        #self.data = []  # removing data that is now saved in file'''

    def draw_image(self, curr_fig=None, create_zoom_butt=True):
        # Note: removing previously plotted plots
        for (
            widget
        ) in (
            self.plotting_frame.winfo_children()
        ):  # FIXME NOTE TODO: USE THIS LATER TO ACCESS BUTTONS FOR MARKING DONE
            # print(f"{widget} --> destroyed")
            widget.destroy()

        if curr_fig is not None:
            self.curr_fig = curr_fig  # NOTE: CURRENT SOLUTION TO TEST FIXME!!

        # showing both raw and sine speed adjusted:
        plt_frame, canvas = self.pack_plot(
            self.plotting_frame, self.curr_fig
        )  # FIXME: or maybe after plotting histo?
        self.curr_canvas = canvas

        if create_zoom_butt:
            self.variable_containers["duplicate"] = {
                "butt": ttk.Button(
                    plt_frame,
                    text="Duplicate",
                    command=self.press_getzoomregion,
                    state="disabled",
                )
            }
            self.variable_containers["duplicate"]["butt"].pack(side="right")

            self.variable_containers["set region"] = {
                "butt": ttk.Button(
                    plt_frame,
                    text="Set Region",
                    command=lambda f=False: self.press_getzoomregion(dup=f),
                )
            }
            self.variable_containers["set region"]["butt"].pack(side="right")

        plt_frame.grid(row=1, column=1, sticky="", padx=0, pady=0)

        # NOTE BELOW IS FAILING:
        # self.logger_box.module_logger.info("Done plotting")

        # for widget in (self.plotting_frame.winfo_children()):  # FIXME NOTE TODO: USE THIS LATER TO ACCESS BUTTONS FOR MARKING DONE
        #    print(f"{widget} --> created")

        # print("-----------")

    def start_scan_widget(self, anal_frm, conf_frm):

        def onclick(event):
            if event.xdata != None and event.ydata != None:
                # print(event.xdata, event.ydata)
                self.x_move_var.set(int(round(event.xdata)))
                self.y_move_var.set(int(round(event.ydata)))

        def press_start():
            if self.demo_mode.get():
                txt = "DEMO/OFFLINE"
            elif self.diagnostics_mode.get():
                txt = "DIAGNOSTICS"
            elif self.record_mode.get():
                txt = "RECORD SCAN"
            else:
                txt = "???"

            self.logger_box.module_logger.info(f"Start scan: {txt}")
            # self.pb['value'] = 0
            self.root.update()  # testing
            self.suggest_name()

            if not self.demo_mode.get():
                folder = self.data_folder.get()
                if len(folder) > 0:
                    if folder[-1] not in ["/", "//", "\\"]:
                        folder += "/"
                self.anal_data_file.set(folder + self.data_file.get())
                self.logger_box.module_logger.info(
                    f"new file name => {self.data_file.get()}"
                )

                if self.speed_mode.get() == "fast":
                    self.t7.fast_galvo_scan()  # NOTE: MUST RECHECK IF THIS STILL WORKS AS INTENDED
                elif self.speed_mode.get() in ["slow", "zoom"]:
                    if self.t7.slow_galvo_scan() is False:
                        return  # scan aborted, do not analyze
                else:
                    print("ERROR: SCAN MODE NOT FOUND")
                    return

                self.logger_box.module_logger.info(f"Done with scan!")

                # NOTE FIXME: MAKE SURE TO WAIT UNTIL DUMPING IS DONE BEFORE ANALYZING
                # CHECK IF WE CAN KNOW WHEN DATAFILE IS READY TO BE USED (CHECK SERVER SIDE)
                self.logger_box.module_logger.info(f"Auto starting analysis of data")

                # windowgui.root.after(int(2 * 1000), windowgui.sleep)
                time.sleep(4)  # TODO REMOVE AND CHANGE TO ANOTHER WAIT/DELAY FUNCTION

                try:
                    press_analyze()
                    self.variable_containers["duplicate"]["butt"].configure(
                        state="normal"
                    )  # activate duplicate button after scan
                except:
                    print("failed analysis after scan")
                    raise

            else:
                self.logger_box.module_logger.info(f"Demo Mode")
                # fig = self.t7.slow_galvo_scan(do_scan=False)
                # self.all_figs = [[fig]]
                # self.draw_image(fig)

            self.root.update()

        def press_cancel():
            print("Pressed cancel")
            self.state.set("cancel")

        def press_close_stream():
            self.t7.close_stream = True

        def press_analyze(from_file=True):

            self.suggest_name()

            self.t7.prev_speed_mode = self.speed_mode.get()
            # TODO NOTE WORKING HERE PLOTTING ANALYSIS
            self.logger_box.module_logger.info("Pressed analyze")

            self.all_figs = self.ETA_analysis()

            if from_file:
                self.t7.slow_galvo_scan(
                    do_scan=False
                )  # Creates scan lists from file vals

            # NOTE: all_figs[0][1] is the matrix with the data
            # self.save_data(data=all_figs[0][1], folder=f"Analysis/{self.find_in_str('_date', self.anal_data_file.get())}", filename=f"{self.find_in_str('_date', self.anal_data_file.get())}_{self.find_in_str('_time', self.anal_data_file.get())}", mode='w')
            # self.zoom_radiobutt.configure(state='normal')  # activating it
            self.draw_image(self.all_figs[0][0])
            cid = self.all_figs[0][0].canvas.mpl_connect(
                "button_press_event", onclick
            )  # NOTE NEW! 23 oct 2024 connect figure to button click

            # DISABLE ALL INPUTS AND BUTTONS ONCE WE ARE DONE

            # self.suggest_name()
            # butt_start.configure(state='disabled')
            # butt_anal.configure(state='disabled')
            # butt_cancel.configure(state='disabled')
            self.slow_radiobutt.configure(state="disabled")
            self.zoom_radiobutt.configure(state="disabled")

            for key in self.variable_containers.keys():
                if "entry" in self.variable_containers[key].keys():
                    self.variable_containers[key]["entry"].configure(
                        state="normal"
                    )  # if disable these, it will only allow duplicate buttom after analysis

                if "butt" in self.variable_containers[key].keys():
                    self.variable_containers[key]["butt"].configure(state="normal")

            self.variable_containers["duplicate"]["butt"].configure(state="normal")

        # frm_send = ttk.Frame(tab, relief=tk.RAISED)

        # ttk.Label(anal_frm, text=f' ').grid(row=1, column=0, sticky="ew")

        butt_start = ttk.Button(conf_frm, text="Start Scan", command=press_start)
        # butt_start = ttk.Button(conf_frm, text="Start Scan", command=press_start, activeforeground='blue', highlightbackground='green')
        butt_cancel = ttk.Button(
            conf_frm, text="Cancel", command=press_cancel
        )  # , activeforeground='blue', highlightbackground='green')
        # butt_close = ttk.Button(conf_frm, text="Close stream", command=press_close_stream)  # , activeforeground='blue', highlightbackground=self.button_color)
        butt_anal = ttk.Button(
            anal_frm, text="Analyze", command=lambda: press_analyze(from_file=True)
        )  # , activeforeground='blue', highlightbackground=self.button_color)

        butt_start.grid(row=50, column=1, columnspan=7, sticky="nsew", padx=0, pady=1.5)
        butt_cancel.grid(
            row=50, column=0, columnspan=1, sticky="nsew", padx=1, pady=1.5
        )
        # butt_close.grid(row=50, column=0, sticky="nsew", padx=0, pady=1.5)
        butt_anal.grid(
            row=50, column=1, columnspan=100, sticky="nsew", padx=1, pady=1.5
        )

        self.variable_containers["start button"] = {"butt": butt_start}
        # self.variable_containers['close stream button'] = {'butt': butt_close}
        self.variable_containers["cancel button"] = {"butt": butt_cancel}
        self.variable_containers["analyze button"] = {"butt": butt_anal}

        # self.pb = ttk.Progressbar(frm_send, style='bar.Horizontal.TProgressbar', orient='horizontal',
        #                          mode='determinate', length=100)  # progressbar
        # self.pb.grid(row=0, column=1, sticky="nsew")

        # return frm_send

    def plot_analysis_widget(self, tab):
        self.plotting_frame = ttk.Frame(tab, relief=tk.RAISED)

        return self.plotting_frame

    def move_with_plot_widget(self, tab):

        def press_moveto():
            self.t7.move_to_pos(
                x_coord=self.y_move_var.get(), y_coord=self.x_move_var.get()
            )  # note: swapped due to camera orientatation

        def press_reset():
            if self.demo_mode.get():
                self.logger_box.module_logger.info("DEMO MODE: WILL NOT CENTER")
                if True:  # input("\n>>> Center anyway?") == "y":
                    try:
                        if not self.t7.handle:
                            self.t7.open_labjack_connection()  # NOTE ONLINE ONLY
                        self.t7.set_offset_pos()
                        # self.t7.close_labjack_connection(closeserver=False)
                        print("Done with centering the laser")
                    except:
                        raise
            else:
                self.logger_box.module_logger.info("Recentered galvo positions")
                self.t7.set_offset_pos()

        frm_move = ttk.Frame(tab, relief=tk.RAISED)

        # ------

        # TODO: hide these until we plot
        move_lab = ttk.Label(frm_move, text=f"Move to pixels (x, y):")
        x_move_entry = ttk.Entry(frm_move, textvariable=self.x_move_var, width=10)
        y_move_entry = ttk.Entry(frm_move, textvariable=self.y_move_var, width=10)
        btn_move = ttk.Button(
            frm_move, text="Move", command=press_moveto
        )  # , activeforeground='blue', highlightbackground=self.button_color)
        btn_reset = ttk.Button(
            frm_move, text="Center/Reset", command=press_reset
        )  # , activeforeground='blue', highlightbackground=self.button_color)

        move_lab.grid(row=0, column=0, columnspan=2, sticky="ew")
        # x_move_entry.grid(row=1, column=0, sticky="ew")
        # y_move_entry.grid(row=1, column=1, sticky="ew")
        # btn_move.grid(row=2, column=0,  columnspan=2, sticky="ew")
        # btn_reset.grid(row=2, column=2,  columnspan=1, sticky="ew")
        x_move_entry.grid(row=1, column=0, sticky="ew", padx=1, pady=1)
        y_move_entry.grid(row=1, column=1, sticky="ew", padx=1, pady=1)
        btn_move.grid(row=1, column=2, columnspan=2, sticky="ew", padx=1, pady=1)
        btn_reset.grid(row=1, column=4, columnspan=1, sticky="ew", padx=1, pady=1)

        idlist = []
        return frm_move

    def set_scope_length_widget(self, tab):
        frm_length = ttk.Frame(tab, relief=tk.RAISED)

        def set_length(event):
            self.ampX.set(round(self.scopelength.get() / self.lensslope.get(), 5))
            self.minX.set(round(-self.ampX.get() + self.t7.x_offset, 10))
            self.maxX.set(round(self.ampX.get() + self.t7.x_offset, 10))

            self.ampY.set(round(self.scopelength.get() / self.lensslope.get(), 5))
            self.minY.set(round(-self.ampY.get() + self.t7.y_offset, 10))
            self.maxY.set(round(self.ampY.get() + self.t7.y_offset, 10))

        def set_length2():
            self.ampX.set(round(self.scopelength.get() / self.lensslope.get(), 5))
            self.minX.set(round(-self.ampX.get() + self.t7.x_offset, 10))
            self.maxX.set(round(self.ampX.get() + self.t7.x_offset, 10))

            self.ampY.set(round(self.scopelength.get() / self.lensslope.get(), 5))
            self.minY.set(round(-self.ampY.get() + self.t7.y_offset, 10))
            self.maxY.set(round(self.ampY.get() + self.t7.y_offset, 10))

        def set_50():
            self.scopelength.set(50)
            set_length2()

        def set_100():
            self.scopelength.set(100)
            set_length2()

        def update():
            self.scopelength.set(self.ampX.get() * self.lensslope.get())

        ttk.Label(
            frm_length, text=f"set scan length um (slope= {self.lensslope.get()})"
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        setlength = ttk.Entry(frm_length, textvariable=self.scopelength, width=12)
        setlength.grid(row=1, column=0, sticky="ew", padx=1, pady=1)
        setlength.bind("<KeyRelease>", set_length)

        set50 = ttk.Button(frm_length, text="set 50", command=set_50)
        set50.grid(row=1, column=1, sticky="ew", padx=1, pady=1)
        set50 = ttk.Button(frm_length, text="set 100", command=set_100)
        set50.grid(row=1, column=2, sticky="ew", padx=1, pady=1)

        return frm_length

    def set_stepsize_widget(self, tab):
        frm_stepsize = ttk.Frame(tab, relief=tk.RAISED)

        def setstep():
            self.dimY.set(round(self.scopelength.get() * 1000 / self.stepsize.get()))

        def getstep():
            self.stepsize.set(round(self.scopelength.get() * 1000 / self.dimY.get(), 5))

        ttk.Label(
            frm_stepsize, text=f"step size in nm. This is only connected to DimY!"
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        step_size_entry = ttk.Entry(frm_stepsize, textvariable=self.stepsize, width=5)
        step_size_entry.grid(row=1, column=0, sticky="ew", padx=1, pady=1)
        ttk.Label(frm_stepsize, text=f"nm").grid(
            row=1, column=1, columnspan=2, sticky="ew"
        )
        step_size_set = ttk.Button(frm_stepsize, text="set", command=setstep)
        step_size_set.grid(row=1, column=2, sticky="ew", padx=1, pady=1)
        step_size_get = ttk.Button(frm_stepsize, text="get", command=getstep)
        step_size_get.grid(row=1, column=3, sticky="ew", padx=1, pady=1)
        return frm_stepsize

    def heatmap_ratio_widget(self, tab):
        frm_ratio = ttk.Frame(tab, relief=tk.RAISED)

        def onclick(event):
            if event.xdata != None and event.ydata != None:
                print(event.xdata, event.ydata)
                self.x_move_var.set(int(round(event.xdata)))
                self.y_move_var.set(int(round(event.ydata)))

        def ratio_plot():
            fig_ratio = draw_image_heatmap_ratio(
                matrix=np.loadtxt("nonspeedmatrix.txt"),
                ratio=self.heatmapratio.get(),
                title=f"ratioplot",
                # f"Theoretical={1/const['scan_fps']} s",
                fig_title=f"ratioplot",
                save_fig=False,
                figsize=(5, 5),
            )
            self.draw_image(fig_ratio)
            cid = fig_ratio.canvas.mpl_connect(
                "button_press_event", onclick
            )  # NOTE NEW! 23 oct 2024 connect figure to button click

        ttk.Label(
            frm_ratio, text=f"set ratio plot(test, can only plot last analyzed file)"
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        heatmap_ratio = ttk.Entry(frm_ratio, textvariable=self.heatmapratio, width=10)
        heatmap_ratio.grid(row=1, column=0, sticky="ew", padx=1, pady=1)
        heatmap_plot = ttk.Button(frm_ratio, text="ratio plot", command=ratio_plot)
        heatmap_plot.grid(row=1, column=1, sticky="ew", padx=1, pady=1)

        return frm_ratio

    def counts_widget(self, tab):
        counts_frame = ttk.Frame(tab)  # always next to tabs (accessible in all tabs)

        def eta_counter_swab(recipe_file, timetag_file, **kwargs):
            # ________ LOAD RECIPE ETA ___________
            with open(recipe_file, "r") as filehandle:
                recipe_obj = json.load(filehandle)
            eta_engine = etabackend.eta.ETA()
            eta_engine.load_recipe(recipe_obj)
            # Set parameters in the recipe
            for arg in kwargs:
                print("Setting", str(kwargs[arg]), "= ", arg)
                eta_engine.recipe.set_parameter(arg, str(kwargs[arg]))

            eta_engine.load_recipe()
            # -------------
            file = Path(timetag_file)
            cutfile = eta_engine.clips(filename=file, format=1)
            result = eta_engine.run(
                {"timetagger1": cutfile}, group="qutag"
            )  # Runs the time tagging analysis and generates histograms

            return result

        def press_count():

            binsize = int(round((1 / (self.freq.get() * 1e-12)) / self.bins.get()))

            # folder = self.data_folder.get()
            # if len(folder) > 0:
            #    if folder[-1] not in ['/', '//', '\\']:
            #        folder += '/'
            file_path = self.anal_data_file.get()  # + folder
            if file_path == "":
                self.logger_box.module_logger.info(
                    "Error: please select a datafile to count!"
                )
                return
            if ".timeres" not in file_path:
                file_path += ".timeres"

            counts = eta_counter_swab(
                "signal_counter.eta", file_path, binsize=binsize, bins=self.bins.get()
            )

            for sigi in signals.keys():
                # if counts[signals[sigi]] > 0:
                counts_labels[sigi].configure(text=f"      {counts[signals[sigi]]}")

        signals = {
            1: "c1",
            2: "c2",
            3: "c3",
            4: "c4",
            # 5: 'c5', 6: 'c6', 7: 'c7',  # 8: 'c8',
            # 100: 'c100', 101: 'c101', 102: 'c102', 103: 'c103',
        }
        counts_labels = {}

        ttk.Label(counts_frame, text=f"Channel", width=7).grid(
            row=1, column=0, sticky="news"
        )  # Channel number
        ttk.Label(counts_frame, text=f"   Counts", width=9, anchor="w").grid(
            row=1, column=1, sticky="news"
        )  # Channel number
        ttk.Button(counts_frame, text=f"  Signal Counter  ", command=press_count).grid(
            row=0, column=0, columnspan=2, sticky="news"
        )  # Channel number

        # Create empty labels (i.e. without any read counts)
        for i, sig in enumerate(signals.keys()):
            ttk.Label(counts_frame, text=f"{sig}", width=6).grid(
                row=i + 2, column=0, sticky="news"
            )  # Channel number
            counts_labels[sig] = ttk.Label(
                counts_frame, text="", width=12, anchor="w"
            )  # Counts
            counts_labels[sig].grid(row=i + 2, column=1, sticky="news")

        return counts_frame

    def log_scan_widget(self, tab):
        frm_log = ttk.Frame(tab, relief=tk.RAISED)

        # returns and grids counts frame
        self.counts_widget(frm_log).grid(
            row=0, column=0, rowspan=1, sticky="news", padx=1, pady=1
        )  # in sub frame

        frm_logbox = ttk.Frame(frm_log, relief=tk.FLAT)
        ttk.Label(frm_logbox, text=f"Log", font=("", 15)).grid(
            row=0, column=0, sticky="w"
        )
        frm_logbox.grid(row=2, column=0, sticky="news", padx=1, pady=1)  # in sub frame
        # TODO: MAKE LOG BOX A DIFFERENT COLOR AND ADD SCROLLBAR
        self.logger_box = Logger(
            frm_logbox, name=self.tabname
        )  # initialize log box from example. note: it grids itself in the class

        # tk.Button(frm_log, text="Disconnect", command=self.t7.close_labjack_connection, activeforeground='blue', highlightbackground=self.button_color).grid(row=3, column=0, sticky='news', padx=0, pady=0)

        return frm_log

    @staticmethod
    def pack_plot(tab, fig, toolbar=True):

        # creating the Tkinter canvas containing the Matplotlib figure
        plt_frame = ttk.Frame(tab, relief=tk.FLAT)
        canvas = FigureCanvasTkAgg(fig, master=plt_frame)  # self.root)
        canvas.draw()

        # placing the canvas on the Tkinter window
        # canvas.get_tk_widget().pack()

        if toolbar:
            # creating the Matplotlib toolbar
            toolbar = NavigationToolbar2Tk(canvas, plt_frame)  # self.root)
            toolbar.update()

        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack(side="top")

        return plt_frame, canvas

    def press_getzoomregion(
        self, dup=True
    ):  # NOTE MUST CALL FUNCTION AGAIN BEFORE SCANNING INCASE SOMETHING IS CHANGED
        print("Pressed get zoom region")
        if True:
            if dup:
                self.nr_of_children += 1
                new_st = windowgui.scan_group.duplicate_scan_tab(
                    tabname=f"{self.tabname}-Z{self.nr_of_children}", copy_tab=self
                )
            else:
                new_st = self

            x_min, x_max = new_st.curr_fig.get_axes()[0].get_xlim()
            y_min, y_max = new_st.curr_fig.get_axes()[
                0
            ].get_ylim()  # NOTE:IMSHOW WILL HAVE Y AXIS FLIPPE #updated1028: is not flipped since the imshow is Ddraw_image_heatmap is origin="low"
            print(
                f"BEFORE ({new_st.curr_fig.get_axes()[0].get_xlim()}), ({new_st.curr_fig.get_axes()[0].get_ylim()})"
            )

            # Indices in the array of voltage values we will take from
            x_min = int(x_min)  # + 0.5)
            x_max = int(x_max)  # - 0.5)
            y_min = int(y_min)  # + 0.5)
            y_max = int(y_max)  # - 0.5)

            print(f"1) ({x_min}, {x_max}), ({y_min}, {y_max})")

            # NOTE THE FIXING BELOW WILL MAKE IT NOT SQUARE ANYMORE
            if True:
                x_min = np.max([0, x_min])
                y_min = np.max([0, y_min])
                x_max = np.min([new_st.t7.step_dim - 1, x_max])
                y_max = np.min([new_st.t7.step_dim - 1, y_max])

            print(f"2) ({x_min}, {x_max}), ({y_min}, {y_max})")

            dx = x_max - x_min
            dy = y_max - y_min

            if dx <= 1:
                print("X PIXEL RANGE TOO SMALL", (x_max - x_min))
                # x_min = np.max([0, x_min-1])   # -= 1
                # x_max = np.min([self.dimY.get(), x_max+1])  # += 1
                x_min -= 1
                x_max += 1
                # TODO NOTE: MAKE SURE WE DON'T GO OUTSIDE ALLOWED RANGE
                print("new x lims:", x_min, x_max)
            if dy <= 1:
                print("Y PIXEL RANGE TOO SMALL", (y_max - y_min))
                # y_min = np.max([0, y_min-1])   # -= 1
                # y_max = np.min([self.dimY.get(), y_max+1])  # += 1
                y_min -= 1
                y_max += 1
                print("new y lims:", y_min, y_max)

            print(f"3) ({x_min}, {x_max}), ({y_min}, {y_max})")

            dx = x_max - x_min
            dy = y_max - y_min
            print("dxy vals:", dx, dy)
            if dx < dy:
                # need to adjust x range
                x_max = x_min + dy
                print("fixed x lims:", x_min, x_max)
            elif dy < dx:
                # need to adjust y range
                y_max = y_min + dx
                print("fixed y lims:", y_min, y_max)
            else:
                pass  # no need to change if already a square

            print(f"4) ({x_min}, {x_max}), ({y_min}, {y_max})")

            dx = x_max - x_min
            dy = y_max - y_min
            print("*dxy vals:", dx, dy)

            # NOTE THE FIXING BELOW WILL MAKE IT NOT SQUARE ANYMORE
            if False:
                x_min = np.max([0, x_min])
                y_min = np.max([0, y_min])
                x_max = np.min([t7.step_dim - 1, x_max])
                y_max = np.min([t7.step_dim - 1, y_max])

            # new_st.curr_fig.get_axes()[0].set_xlim(x_min - 0.5, x_max + 0.5)
            # new_st.curr_fig.get_axes()[0].set_ylim(y_max + 0.5, y_min - 0.5)  # NOTE FLIPPED AXIS

            if dup:
                self.curr_fig.get_axes()[0].set_xlim(x_min, x_max)
                self.curr_fig.get_axes()[0].set_ylim(y_min, y_max)  # NOTE FLIPPED AXIS
                self.curr_canvas.draw()

            new_st.curr_fig.get_axes()[0].set_xlim(x_min, x_max)
            new_st.curr_fig.get_axes()[0].set_ylim(y_min, y_max)  # NOTE FLIPPED AXIS
            new_st.curr_canvas.draw()

            print(f"AFTER ({x_min}, {x_max}), ({y_min}, {y_max})")
            print(
                f"AFTER ({new_st.curr_fig.get_axes()[0].get_xlim()}), ({new_st.curr_fig.get_axes()[0].get_ylim()})"
            )
            print("------")

            # ---- SETTING NEW PARAMS FOR NEW SCAN
            # self.clue
            # self.bins
            # self.int_time = tk.DoubleVar(value=1.0)
            # self.freq = tk.DoubleVar(value=1.0)
            # self.speed_mode
            # self.data_file

            # self.prev_dimY = tk.IntVar(value=self.dimY.get())   # NOTE WARNING MAYBE WE SHOULD GET IT FROM THE LOADED FILENAME INSTEAD
            # self.prev_ampX = tk.DoubleVar(value=self.ampX.get())
            # self.prev_ampY = tk.DoubleVar(value=self.ampY.get())

            new_st.t7.find_zoom_range(
                y_min, y_max, x_min, x_max
            )  # try to get voltages note: swapped due to camera orientatation
            new_st.select_speed()

        """else:
            self.speed_mode.set('zoom')

            x_min, x_max = self.curr_fig.get_axes()[0].get_xlim()
            y_max, y_min = self.curr_fig.get_axes()[0].get_ylim()  # NOTE:IMSHOW WILL HAVE Y AXIS FLIPPED

            print(f"BEFORE ({self.curr_fig.get_axes()[0].get_xlim()}), ({self.curr_fig.get_axes()[0].get_ylim()})")

            x_min = int(x_min+0.5)
            x_max = int(x_max-0.5)
            y_min = int(y_min+0.5)
            y_max = int(y_max-0.5)

            print(f"1) ({x_min}, {x_max}), ({y_min}, {y_max})")

            # NOTE THE FIXING BELOW WILL MAKE IT NOT SQUARE ANYMORE
            if True:
                x_min = np.max([0, x_min])
                y_min = np.max([0, y_min])
                x_max = np.min([self.t7.step_dim-1, x_max])
                y_max = np.min([self.t7.step_dim-1, y_max])

            print(f"2) ({x_min}, {x_max}), ({y_min}, {y_max})")

            dx = x_max - x_min
            dy = y_max - y_min

            if dx <= 1:
                print("X PIXEL RANGE TOO SMALL", (x_max-x_min))
                #x_min = np.max([0, x_min-1])   # -= 1
                #x_max = np.min([self.dimY.get(), x_max+1])  # += 1
                x_min -= 1
                x_max += 1
                # TODO NOTE: MAKE SURE WE DON'T GO OUTSIDE ALLOWED RANGE
                print("new x lims:", x_min, x_max)
            if dy <= 1:
                print("Y PIXEL RANGE TOO SMALL", (y_max-y_min))
                #y_min = np.max([0, y_min-1])   # -= 1
                #y_max = np.min([self.dimY.get(), y_max+1])  # += 1
                y_min -= 1
                y_max += 1
                print("new y lims:", y_min, y_max)

            print(f"3) ({x_min}, {x_max}), ({y_min}, {y_max})")

            dx = x_max - x_min
            dy = y_max - y_min
            print("dxy vals:", dx, dy)
            if dx < dy:
                # need to adjust x range
                x_max = x_min + dy
                print("fixed x lims:", x_min, x_max)
            elif dy < dx:
                # need to adjust y range
                y_max = y_min + dx
                print("fixed y lims:", y_min, y_max)
            else:
                pass  # no need to change if already a square

            print(f"4) ({x_min}, {x_max}), ({y_min}, {y_max})")

            dx = x_max - x_min
            dy = y_max - y_min
            print("*dxy vals:", dx, dy)

            # NOTE THE FIXING BELOW WILL MAKE IT NOT SQUARE ANYMORE
            if False:
                x_min = np.max([0, x_min])
                y_min = np.max([0, y_min])
                x_max = np.min([t7.step_dim-1, x_max])
                y_max = np.min([t7.step_dim-1, y_max])

            self.curr_fig.get_axes()[0].set_xlim(x_min-0.5, x_max+0.5)
            self.curr_fig.get_axes()[0].set_ylim(y_max+0.5, y_min-0.5)   # NOTE FLIPPED AXIS

            self.curr_canvas.draw()

            print(f"AFTER ({x_min}, {x_max}), ({y_min}, {y_max})")
            print(f"AFTER ({self.curr_fig.get_axes()[0].get_xlim()}), ({self.curr_fig.get_axes()[0].get_ylim()})")
            print("------")

            # ---- SETTING NEW PARAMS FOR NEW SCAN
            #self.clue
            #self.bins
            #self.int_time = tk.DoubleVar(value=1.0)
            #self.freq = tk.DoubleVar(value=1.0)
            #self.speed_mode
            #self.data_file

            #self.prev_dimY = tk.IntVar(value=self.dimY.get())   # NOTE WARNING MAYBE WE SHOULD GET IT FROM THE LOADED FILENAME INSTEAD
            #self.prev_ampX = tk.DoubleVar(value=self.ampX.get())
            #self.prev_ampY = tk.DoubleVar(value=self.ampY.get())

            self.t7.find_zoom_range(x_min, x_max, y_min, y_max)  # try to get voltages

            self.select_speed()"""

    def close(self, printlog):

        if printlog:
            self.logger_box.module_logger.info("TODO: IMPLEMENT SOMETHING WHEN CLOSING")
        else:
            print("*** TODO: IMPLEMENT SOMETHING WHEN CLOSING")
        # self.sq.websq_disconnect()  # close SQWeb connection

    def ETA_analysis(self):
        # ------IMPORTS-----
        import Swabian_Microscope_library_copy1028 as Q

        # ------------ PARAMETERS AND CONSTANTS --------------
        timetag_file = self.anal_data_file.get()  # + folder
        if ".timeres" not in timetag_file:
            timetag_file += ".timeres"

        # FIGURING OUT IF FAST MODE OR SLOW MODE
        try:
            try:  # TESTING FOR FAST MODE
                eval(self.find_in_str("_sineFreq", timetag_file))
                speed_mode = "fast"
                self.logger_box.module_logger.info("NOTE: FAST FOUND IN FILE NAME")
            except:  # TESTING FOR SLOW MODE
                try:
                    eval(self.find_in_str("_dwellTime", timetag_file))
                    speed_mode = "slow"
                    # self.logger_box.module_logger.info("NOTE: SLOW FOUND IN FILE NAME")
                except:
                    try:
                        eval(self.find_in_str("_intTime", timetag_file))
                        speed_mode = "slow"
                        self.logger_box.module_logger.info(
                            "NOTE: SLOW FOUND IN FILE NAME"
                        )
                    except:
                        self.logger_box.module_logger.info(
                            f"ERROR: FAILED TO FIND HINTS FOR SPEED MODE in: {timetag_file}"
                        )
                        return
            self.speed_mode.set(speed_mode)
            self.select_speed()
        except:  # NOTE TODO CHECK: UNSURE IF THIS EVER RUNS
            self.logger_box.module_logger.info(
                "***ERROR: SPEEDMODE SEARCH FAILED, set to fast"
            )
            speed_mode = "fast"
            self.speed_mode.set(speed_mode)

        eta_recipe = (
            self.eta_recipe.get()
        )  # 'Swabian_multiframe_recipe_bidirectional_segments_marker4_20.eta'  # 'microscope_bidirectional_segments_0.0.3.eta'
        clue = (
            self.clue.get()
        )  # Note: this is used to help find the correct timeres file when only given frequency (ex: 'higher_power', 'digit_8', '13h44m23s')
        ch_sel = self.ch_sel.get()
        sweep_mode = self.sweep_mode.get()  # linear
        folder = self.data_folder.get()

        # NUM FRAME
        try:
            nr_frames = eval(self.find_in_str("_numFrames", timetag_file))
            self.nr_frames.set(nr_frames)
        except:
            nr_frames = 1  # self.nr_frames.get()
            # self.logger_box.module_logger.info("NOTE: Nr frames is set to 1")

        # SINE X AMP
        try:
            ampX = eval(self.find_in_str("_sineAmp", timetag_file))
            self.ampX.set(ampX)  # --> step values between -0.3 and 0.3
            minX = round(-ampX + self.t7.x_offset, 10)
            maxX = round(ampX + self.t7.x_offset, 10)
            self.minX.set(minX)
            self.maxX.set(maxX)

        except:
            try:
                ampX = eval(self.find_in_str("_xAmp", timetag_file))
                self.ampX.set(ampX)  # --> step values between -0.3 and 0.3
                minX = round(-ampX + self.t7.x_offset, 10)
                maxX = round(ampX + self.t7.x_offset, 10)
                self.minX.set(minX)
                self.maxX.set(maxX)
            except:
                try:
                    minX = eval(self.find_in_str("_xMin", timetag_file))
                    maxX = eval(self.find_in_str("_xMax", timetag_file))
                    self.minX.set(minX)
                    self.maxX.set(maxX)

                    ampX = round((maxX - minX) / 2, 10)
                    self.ampX.set(ampX)  # --> sine values between -0.3 and 0.3

                except:
                    ampX = self.ampX.get()
                    self.logger_box.module_logger.info(
                        "WARNING: Horizontal (x) amplitude fetched from GUI input"
                    )
                    self.logger_box.module_logger.info(
                        "WARNING: min, max of X fetched from GUI input"
                    )
                    minX = self.minX.get()  # TODO: MAYBE CALCULATE INSTEAD??
                    maxX = self.maxX.get()  # TODO: MAYBE CALCULATE INSTEAD??
                    raise
                    return

        # STEP Y AMP
        try:
            ampY = eval(self.find_in_str("_stepAmp", timetag_file))
            self.ampY.set(ampY)  # --> sine values between -0.3 and 0.3
            minY = round(-ampY + self.t7.y_offset, 10)
            maxY = round(ampY + self.t7.y_offset, 10)
            self.minY.set(minY)
            self.maxY.set(maxY)

        except:
            try:
                ampY = eval(self.find_in_str("_yAmp", timetag_file))
                self.ampY.set(ampY)  # --> sine values between -0.3 and 0.3
                minY = round(-ampY + self.t7.y_offset, 10)
                maxY = round(ampY + self.t7.y_offset, 10)
                self.minY.set(minY)
                self.maxY.set(maxY)
            except:
                try:
                    minY = eval(self.find_in_str("_yMin", timetag_file))
                    maxY = eval(self.find_in_str("_yMax", timetag_file))
                    self.minY.set(minY)
                    self.maxY.set(maxY)

                    ampY = round((maxY - minY) / 2, 10)
                    self.ampY.set(ampY)  # --> sine values between -0.3 and 0.3
                except:
                    ampY = self.ampY.get()
                    minY = self.minY.get()  # TODO: MAYBE CALCULATE INSTEAD??
                    maxY = self.maxY.get()  # TODO: MAYBE CALCULATE INSTEAD??
                    self.logger_box.module_logger.info(
                        "WARNING: min, max of Y fetched from GUI input"
                    )
                    self.logger_box.module_logger.info(
                        "WARNING: Vertical (y) amplitude fetched from GUI input"
                    )
                    raise
                    return

        # STEP DIM (XY DIM)
        try:
            dimX = eval(self.find_in_str("_stepDim", timetag_file))
            self.dimY.set(dimX)
            dimY = dimX  # = int(round(dimX * (ampY / ampX)))
        except:
            try:
                dimX = eval(self.find_in_str("_xyDim", timetag_file))
                self.dimY.set(dimX)
                dimY = dimX
            except:
                dimX = (
                    self.dimY.get()
                )  # how many (stepwise) steps we take in scan       # TODO FIXME: be consistent with X and Y naming
                self.logger_box.module_logger.info(
                    "WARNING: Step dim fetched from GUI input"
                )
                dimY = dimX

        # INT TIME or FREQ
        if speed_mode == "fast":  # FAST MODE
            # SINE FREQ
            try:
                freq = eval(self.find_in_str("_sineFreq", timetag_file))
                self.freq.set(freq)  # 10
            except:
                freq = self.freq.get()  # 10
                self.logger_box.module_logger.info(
                    "WARNING: FAST MODE ACTIVE BUT FREQUENCY IS MISSING FROM FILE NAME \n    |> Freq fetched from GUI input"
                )

            int_time = 1 / (freq * dimX)

            # bins = self.bins.get()  # how many bins/containers we get back for one period   #20000 is good --> 10k per row
            bins = int(2 * dimX)
            self.bins.set(bins)
        elif speed_mode in ["slow", "zoom"]:
            # INT TIME or DWELL TIME
            try:
                int_time = eval(self.find_in_str("_intTime", timetag_file))
                self.int_time.set(int_time)  # 10
            except:
                try:
                    int_time = eval(self.find_in_str("_dwellTime", timetag_file))
                    self.int_time.set(int_time)  # 10
                except:
                    int_time = self.int_time.get()  # 10
                    self.logger_box.module_logger.info(
                        "WARNING: SLOW MODE ACTIVE BUT INTEGRATION TIME IS MISSING FROM FILE NAME"
                    )
                    self.logger_box.module_logger.info(
                        "--> Integration time fetched from GUI input"
                    )

            freq = 1 / (int_time * dimX)  # FIXME NOTE: NOT USED!!! MAYBE REMOVE

            # bins = self.bins.get()  # how many bins/containers we get back for one period   #20000 is good --> 10k per row
            bins = int(dimX)  # note: in slow mode we only sweep row once, not twice
            self.bins.set(bins)
        else:
            self.logger_box.module_logger.info(
                f"ERROR: speed mode '{speed_mode}' not recognized"
            )
            return

        # SCANTIME
        try:
            scantime = eval(self.find_in_str("_scantime", timetag_file))
        except:
            if speed_mode == "fast":
                scantime = round(dimX / freq, 2)
            else:
                scantime = round(int_time * dimX * dimY, 2)
            self.logger_box.module_logger.info(
                "WARNING: Scan time calculated from GUI input"
            )

        """# Playback Frame Rates for GIFs:
        # (time for one frame) = (number of steps)*(period) = (dimX)/(frequency)
        #scan_fps = freq / dimX  # frame rate = (1/(time for one frame)) = (freq/dimX)
        #gif_rates = [1, 5, scan_fps]  # playback frame rates for each gif we want to create
        #gif_notes = ["", "", "(live)"]  # notes for gif we want (for each playback frame rate)
        """
        # ------------ MISC. RELATIONSHIPS ------------
        freq_ps = freq * 1e-12  # frequency scaled to unit picoseconds (ps)
        period_ps = 1 / freq_ps  # period in unit picoseconds (ps)
        binsize = int(
            round(period_ps / bins)
        )  # how much time (in ps) each histogram bin is integrated over (=width of bins). Note that the current recipe returns "bins" values per period.
        # ^ NOTE FIXME TODO: DOES THE ABOVE NEED TO BE DIFFERENT FRO SLOW MODE?

        # self.logger_box.module_logger.info(f"bins = {bins}\nbinsize = {binsize}")  # *{10e-12} picoseconds")

        # NOTE: Below is a dictionary with all the parameters defined above. This way we can sent a dict with full access instead of individual arguments
        const = {
            "eta_recipe": eta_recipe,
            "timetag_file": timetag_file,
            "clue": clue,
            "folder": folder,
            "nr_frames": nr_frames,
            "freq": freq,
            "ampX": ampX,
            "ampY": ampY,
            "dimX": dimX,
            "dimY": dimY,
            "minX": minX,  # * !
            "maxX": maxX,  # * !
            "minY": minY,  # * !
            "maxY": maxY,  # * !
            "bins": bins,
            "ch_sel": ch_sel,
            "freq_ps": freq_ps,
            "period_ps": period_ps,
            "binsize": binsize,
            # "scan_fps": scan_fps,
            # "gif_rates": gif_rates,
            # "gif_notes": gif_notes,
            "sweep_mode": sweep_mode,
            "speed_mode": speed_mode,
            "scantime": scantime,  # TODO FIXME!!!
        }

        self.suggest_name()

        # print("Using recipe:", eta_recipe)
        # --------- GET DATA AND HISTOGRAMS------------

        # quick version of "ad infinitum" code where we generate one image/frame at a time from ETA

        # --- GET TIMETAG FILE NAME, unless manually provided ---
        if timetag_file is None:
            timetag_file = Q.get_timres_name(folder, nr_frames, freq, clue=clue)
            const["timetag_file"] = timetag_file
        self.logger_box.module_logger.info(f"Using datafile: {timetag_file}\n")
        self.root.update()

        # --- PROVIDE WHICH MAIN FOLDER WE SAVE ANY ANALYSIS TO (ex. images, raw data files, etc.), DEPENDING ON WHICH ETA FILE
        st = timetag_file.find("date(")
        fin = timetag_file.find(".timeres")
        const["save_location"] = (
            f"Analysis/{timetag_file[st:fin]}"  # This is the folder name for the folder where data, images, and anything else saved from analysis will be saved
        )

        #       FOR EXAMPLE: const["save_location"] = Analysis/(100Hz)_date(230717)_time(14h02m31s)

        # --- EXTRACT AND ANALYZE DATA ---
        # all_figs1 = Q.eta_segmented_analysis_multiframe(const=const)   # note: all params we need are sent in with a dictionary. makes code cleaner

        # testing prev version below

        self.t7.get_scan_parameters()  # NEWWNEWW ! Fetch new params
        all_figs2 = Q.bap_eta_segmented_analysis_multiframe(
            const=const
        )  # note: all params we need are sent in with a dictionary. makes code cleaner
        # NOTE: july 2024, removing sine adjusted because we don't want to do sine movement anymore
        return all_figs2

    @staticmethod
    def find_in_str(term, search_str):
        try:
            return re.search(f"{term}\((.*?)\)", search_str).group(1)
        except:
            print(f"ERROR: Failed string search for '{term}' in '{search_str}'")
            return -1.0


class NewScanTabGroup:  # NEW SCANS
    def __init__(self, parent_tab, tab_name=""):

        self.parent_tab = parent_tab
        self.tab_counter = 0

        self.root_nb = NBControl.add_notebook(
            parent_tab=parent_tab, expand=1, fill="both", side="right"
        )

        self.root_tab = NBControl.add_tab(
            parent_nb=self.root_nb, tab_name=tab_name
        )  # INITIAL SCAN TAB
        _ = ScanTab(self.root_tab, tab_name)

        self.tab_counter += 1

        self.plus_tab = NBControl.add_tab(parent_nb=self.root_nb, tab_name="+")
        ttk.Button(
            self.plus_tab, text="Add New Scan", command=self.append_scan_tab
        ).grid(
            row=0, column=0
        )  # FOR ADDING MORE TABS,... activeforeground='blue'

        # tk.Button(self.root_tab, text="+zoom", activeforeground='blue', command=self.create_child_scangroup).grid(row=0, column=0)  # FOR ADDING MORE TABS

    def append_scan_tab(self, tabname=None):
        self.tab_counter += 1
        if tabname:
            tab = NBControl.add_tab(
                parent_nb=self.root_nb, tab_name=tabname
            )  # ADD EXTRA SCANS
        else:
            tabname = f"S{self.tab_counter}"
            tab = NBControl.add_tab(
                parent_nb=self.root_nb, tab_name=tabname
            )  # ADD EXTRA SCANS
        st = ScanTab(tab, tabname)

        return st

    def duplicate_scan_tab(self, tabname=None, copy_tab=None):

        if not tabname:
            tabname = f"selectname"
            # Note: we probably don't want the above to happen
            print("ERROR IN DUPLICATE TAB")
            raise

        if copy_tab:
            # copy_tab.nr_of_children += 1

            tab = NBControl.add_tab(
                parent_nb=self.root_nb, tab_name=tabname
            )  # , child_tab=copy_tab)  # ADD EXTRA SCANS

            if (
                True
            ):  # TODO: MANUALLY MAKE A DEEP COPY OF EVERYTHING BESIDES TKINTER OBJECTS

                st = ScanTab(tab, tabname)

                st.parent = copy_tab  # check if we use this and/or need to use this

                # ----- TODO: COPY OVER and IMAGE

                """if False:
                    print("SCANTAB ATTRIBUTES:")
                    copy_attr = copy_tab.__dict__   #copy_attr = vars(copy_tab) ?
                    st_attr = st.__dict__    # copy_attr = vars(copy_tab) ?

                    failed_attr = []
                    failed_types = []
                    missing_attr = []
                    for key in copy_attr.keys():
                        #print(key, "-->", copy_attr[key])

                        if key in st_attr.keys():
                            try:
                                st.__dict__[key] = copy.deepcopy(copy_attr[key])
                            except:
                                #print("FAILED TO COPY:", key, "of type:", type(copy_attr[key]))
                                attr_type = str(type(copy_attr[key]))
                                if attr_type not in failed_types:
                                    failed_types.append(attr_type)
                                failed_attr.append(key)

                                if attr_type == "<class 'tkinter.ttk.Frame'>":
                                    pass

                                elif attr_type == "<class 'tkinter.StringVar'>":
                                    st.__dict__[key] = tk.StringVar(value=copy_attr[key].get())

                                elif attr_type == "<class 'dict'>":
                                    pass

                                elif attr_type == "<class '__main__.Logger'>":
                                    pass   # no need to fix

                                elif attr_type == "<class 'tkinter.IntVar'>":
                                    st.__dict__[key] = tk.IntVar(value=copy_attr[key].get())

                                elif attr_type == "<class 'tkinter.DoubleVar'>":
                                    st.__dict__[key] = tk.DoubleVar(value=copy_attr[key].get())

                                elif attr_type == "<class 'tkinter.Radiobutton'>":
                                    pass

                                elif attr_type == "<class 'tkinter.BooleanVar'>":
                                    st.__dict__[key] = tk.BooleanVar(value=copy_attr[key].get())

                                elif attr_type == "<class 'tkinter.Frame'>":
                                    pass

                                elif attr_type == "<class '__main__.T7'>":
                                    pass

                                else:
                                    print("ERROR: COULD NOT FIND TYPE TO COPY")
                        else:
                            #print(key, "not in new object dict")
                            missing_attr.append(key)
                        #print(type(copy_attr[key]))

                    print("failed_types", failed_types)
                    print("failed_attr", failed_attr)
                    print("missing_attr", missing_attr)
                else:"""
                st.state.set(copy_tab.state.get())
                st.speed_mode.set(copy_tab.speed_mode.get())
                # st.params = copy_tab.params
                st.clue.set(f"({tabname})_" + copy_tab.clue.get())
                st.dimX.set(copy_tab.dimX.get())
                st.dimY.set(copy_tab.dimY.get())
                st.int_time.set(copy_tab.int_time.get())
                st.ampX.set(copy_tab.ampX.get())
                st.ampY.set(copy_tab.ampY.get())
                st.minX.set(copy_tab.minX.get())
                st.minY.set(copy_tab.minY.get())
                st.maxX.set(copy_tab.maxX.get())
                st.maxY.set(copy_tab.maxY.get())
                st.speed_mode.set(copy_tab.speed_mode.get())
                st.scopelength.set(st.ampX.get() * copy_tab.lensslope.get())
                # st.variable_containers.set(copy_tab.variable_containers.get())

                # ------
                st.all_figs = copy.deepcopy(copy_tab.all_figs)
                st.curr_fig = copy.deepcopy(copy_tab.curr_fig)
                # st.draw_image(create_zoom_butt=False)
                st.draw_image(create_zoom_butt=True)

                st.t7.get_scan_parameters_copy(parent=copy_tab.t7)
                st.zoom_radiobutt.configure(state="normal")  # activating it
                st.slow_radiobutt.configure(state="disabled")  # activating it
                st.speed_mode.set("zoom")  # this is causing problems
                st.suggest_name()
                # -------
                # st.draw_image(st.all_figs[0][0])

            else:
                st = copy.copy(copy_tab)

                st.tabname = tabname
                st.tab = tab
                st.nr_of_children = 0
                # tk.Button(tab, text="Destroy Tab", activeforeground='blue', background='red', command=st.destroy_tab).grid(row=0, column=0, sticky='w')  # FOR ADDING MORE TABS
                ttk.Button(tab, text="Destroy Tab", command=st.destroy_tab).grid(
                    row=1, column=0, sticky="ws"
                )  # FOR ADDING MORE TABS
                st.init_fill_tabs()
                st.draw_image(st.all_figs[0][0])
                st.zoom_radiobutt.configure(state="normal")  # activating it
                st.slow_radiobutt.configure(state="disabled")  # activating it

        else:
            # Add a fresh and new scan tab
            tab = NBControl.add_tab(
                parent_nb=self.root_nb, tab_name=tabname
            )  # ADD EXTRA SCANS
            st = ScanTab(tab, tabname)

        return st


class WindowGUI:

    def __init__(self):
        # Create an empty window
        self.root = self.init_window()

        # -------

        # Create a notebook for tabs!
        self.root_nb = NBControl.add_notebook(
            parent_tab=self.root, expand=1, fill="both", side="right"
        )

        # Add scan tab! (Note: later we can have a settings tab)
        self.scans_tab = NBControl.add_tab(parent_nb=self.root_nb, tab_name="Scans")
        self.settings_tab = NBControl.add_tab(
            parent_nb=self.root_nb, tab_name="Settings"
        )

        # -------

        # Add a scan tab and a plus button
        self.scan_group = NewScanTabGroup(parent_tab=self.scans_tab, tab_name="S1")

    @staticmethod
    def init_window():
        try:
            # from ttkthemes import ThemedTk  # NOTE FIGURE OUT

            root = ThemedTk(theme="yaru")  # 'yaru' , 'radiance', 'breeze'
        except:
            print("Warning: Could not use ThemedTk(), using Tk() instead.")
            root = tk.Tk()

        root.title("Quantum Microscope GUI")  # *Ghostly matters*
        root.resizable(True, True)
        root.geometry("1200x900")
        return root

    def sleep(self):
        # empty function used instead of time.sleep
        # use by writing: "windowgui.root.after(int(time*1000), windowgui.sleep)"
        return


class Logger:  # example from: https://stackoverflow.com/questions/30266431/create-a-log-box-with-tkinter-text-widget
    # TODO: add clear button for logger

    # this item "module_logger" is visible only in this module,
    # (but you can create references to the same logger object from other modules
    # by calling getLogger with an argument equal to the name of this module)
    # this way, you can share or isolate loggers as desired across modules and across threads
    # ...so it is module-level logging and it takes the name of this module (by using __name__)
    # recommended per https://docs.python.org/2/library/logging.html

    def __init__(self, tk_window, name=__name__):
        # self.module_logger = logging.getLogger(__name__)
        # print("NEW LOGGER --> ", __name__, "--> ", logging.getLogger(__name__))

        self.module_logger = logging.getLogger(name)
        print("NEW LOGGER --> ", name, "--> ", logging.getLogger(name))

        # create Tk object instance
        app = self.simpleapp_tk(tk_window)

        # setup logging handlers using the Tk instance created above the pattern below can be used in other threads...
        #   to allow other thread to send msgs to the gui
        # in this example, we set up two handlers just for demonstration (you could add a fileHandler, etc)
        stderrHandler = logging.StreamHandler()  # no arguments => stderr
        self.module_logger.addHandler(stderrHandler)
        guiHandler = self.MyHandlerText(app.mytext)
        self.module_logger.addHandler(guiHandler)
        self.module_logger.setLevel(logging.INFO)

        # NOTE THIS IS HOW YOU LOG INTO THE BOX:
        # self.module_logger.info("...some log text...")

    def simpleapp_tk(self, parent):
        # tk.Tk.__init__(self, parent)
        self.parent = parent

        # self.grid()
        # self.mybutton = ttk.Button(parent, text="...")
        # self.mybutton.grid(row=0, column=1, sticky='w')
        # self.mybutton.bind("<ButtonRelease-1>", self.button_callback)

        """self.mybutton = ttk.Button(parent, text="Clear")
        self.mybutton.grid(row=0, column=2, sticky='e')
        self.mybutton.bind("<ButtonRelease-1>", self.clear_button_callback)"""

        self.mytext = tk.Text(
            parent,
            state="disabled",
            height=30,
            width=25,
            wrap="word",
            background="#eeeeee",
        )
        self.mytext.grid(row=1, column=0, columnspan=3)

        return self

    def button_callback(self, event):
        now = time.strftime("%H:%M:%S", time.localtime())
        msg = "hai!"
        self.module_logger.info(f"({now})  ->  {msg}")

    class MyHandlerText(logging.StreamHandler):
        def __init__(self, textctrl):
            logging.StreamHandler.__init__(self)  # initialize parent
            self.textctrl = textctrl

        def emit(self, record):
            msg = f'-- [{time.strftime("%H:%M:%S", time.localtime())}]\n' + self.format(
                record
            )
            self.textctrl.config(state="normal")
            self.textctrl.insert("end", msg + "\n\n")
            self.flush()
            self.textctrl.config(state="disabled")
            self.textctrl.see("end")


class T7:

    def __init__(self, master):
        self.logger_box = master.logger_box
        self.safetytests = SafetyTests(master, self)
        self.gui = master

        self.handle = None  # Labjack device handle
        self.abort_scan = False  # Important safety bool for parameter checks
        self.close_stream = False

        # --------------- HARDCODED CLASS CONSTANTS BASED ON WIRING -------------

        """trigger  marker        wait          marker   y-addr   marker      wait           marker  trigger  stream on                                                                                stream off
        [     'FIO0', 'FIO5', 'WAIT_US_BLOCKING', 'FIO5', 'TDAC2','FIO5', 'WAIT_US_BLOCKING', 'FIO5', 'FIO0', 'WAIT_US_BLOCKING', 'WAIT_US_BLOCKING', 'WAIT_US_BLOCKING', 

        'STREAM_ENABLE', 'FIO0', 'STREAM_NUM_SCANS', 'STREAM_ENABLE', 'FIO5', 'WAIT_US_BLOCKING', 'FIO5', 'TDAC2', 'FIO5', 'WAIT_US_BLOCKING', 'FIO5', 'FIO0', 'WAIT_US_BLOCKING', 'WAIT_US_BLOCKING', 'WAIT_US_BLOCKING', 
        'STREAM_ENABLE', 'FIO0', 'STREAM_NUM_SCANS', 'STREAM_ENABLE', 'FIO5', 'WAIT_US_BLOCKING', 'FIO5', 'TDAC2', 'FIO5', 'WAIT_US_BLOCKING', 'FIO5', 'FIO0', 'WAIT_US_BLOCKING', 'WAIT_US_BLOCKING', 'WAIT_US_BLOCKING', 
        'STREAM_ENABLE', 'FIO0', 'STREAM_NUM_SCANS', 'STREAM_ENABLE', 'FIO5', 'WAIT_US_BLOCKING', 'FIO5', 'TDAC2', 'FIO5', 'WAIT_US_BLOCKING', 'FIO5', 'FIO0', 'WAIT_US_BLOCKING', 'WAIT_US_BLOCKING', 'WAIT_US_BLOCKING', 
        ...
        """

        self.wait_address = "WAIT_US_BLOCKING"
        self.x_address = "DAC1"  # Values sent from periodic buffer (which is not compatible with TDAC) this is Xaxis which does sweep
        self.y_address = (
            "TDAC2"  # TickDAC via LJ port "FIO2" (TDAC IN PORTS FIO2 FIO3) Yaxis
        )

        # as of november 2023, changed wiring to FIO5 (with coaxial) Talks to Timetagger
        self.q_M101_addr = "FIO5"  # ch2?
        self.q_M102_addr = "FIO5"  # ch2?

        # TRIGGERED STREAM, USING FIO0 and FIO1: (Talks to itself) to trigger
        self.tr_source_addr = (
            "FIO0"  # Address for channel that outputs the trigger pulse
        )
        self.tr_sink_addr = "FIO1"  # Address for channel that gets trigger pulse, and trigger stream on/off when pulse is recieved

        # Physical offset due to linearization of system (units: volts)
        self.x_offset = 0.59
        self.y_offset = -0.289

    def move_to_pos(self, x_coord, y_coord):

        try:
            if self.gui.demo_mode.get():
                self.get_scan_parameters()

            x_vals = np.around(
                np.linspace(
                    start=self.minX,
                    stop=self.maxX,
                    num=int(self.step_dim),
                    endpoint=True,
                ),
                decimals=10,
            )
            y_vals = np.around(
                np.linspace(
                    start=self.minY,
                    stop=self.maxY,
                    num=int(self.step_dim),
                    endpoint=True,
                ),
                decimals=10,
            )

            """if self.speed_mode in ['fast', 'slow']:
                print("WARNING: FINDING PIXEL POS FROM FAST OR SLOW RANGE")
                # RECREATING SCAN VALUES
                x_vals = np.around(np.linspace(start=-self.sine_amp, stop=self.sine_amp, num=int(self.step_dim), endpoint=True) + self.x_offset, decimals=10)
                y_vals = np.around(np.linspace(start=-self.step_amp, stop=self.step_amp, num=int(self.step_dim), endpoint=True) + self.y_offset, decimals=10)

            elif self.speed_mode == 'zoom':
                print("WARNING: FINDING PIXEL ZOOM FROM ZOOM RANGE")
                x_vals = np.around(np.linspace(start=self.minX, stop=self.maxX, num=int(self.step_dim), endpoint=True), decimals=10)
                y_vals = np.around(np.linspace(start=self.minY, stop=self.maxY, num=int(self.step_dim), endpoint=True), decimals=10)
            else:
                print("ERROR IN MOVE TO POS: MODE NOT FOUND")
                return"""
            if not (0 <= x_coord < self.step_dim):
                self.logger_box.module_logger.info(
                    f"Aborted moving: Coordinates ({x_coord}, {y_coord}) must be in range ({0}, {self.step_dim-1}) "
                )
                return
            if not (0 <= y_coord < self.step_dim):
                self.logger_box.module_logger.info(
                    f"Aborted moving: Coordinates ({x_coord}, {y_coord}) must be in pixel range ({0}, {self.step_dim-1}) "
                )
                return

            x_pos = x_vals[x_coord]
            y_pos = y_vals[y_coord]

            self.logger_box.module_logger.info(
                f"Chosen voltages: ({x_pos}, {y_pos}) at pixel ({x_coord}, {y_coord})"
            )

        except:
            self.logger_box.module_logger.info("Error getting move vars")
            raise
            return

        # TODO: ADD SAFETY CHECK BEFORE TRYING
        # ------------------------------------------
        abort_move = False

        if self.gui.demo_mode.get():
            self.logger_box.module_logger.info("DEMO MODE: WILL NOT MOVE")

            if True:  # input("\n>>> Do you want to move anyway?\n") == 'y':
                pass
            else:
                abort_move = True

        # ----------- STEP BETWEEN ROWS (Y) -----------
        # CHECKING STEP INPUT VALUES TO SERVO, MAX ALLOWED IS 5V, WE HAVE 4V FOR MARGINS
        if abs(y_pos) > 4:
            self.logger_box.module_logger.info(
                f"Error: Too large voltage ({y_pos}V) provided for Y (step)!"
            )
            abort_move = True

        # ----------- STEP WITHIN ROW (X) -----------
        # CHECKING SINE INPUT VALUES TO SERVO, MAX ALLOWED IS 5V, WE HAVE 4V FOR MARGINS
        if abs(x_pos) > 4:
            self.logger_box.module_logger.info(
                f"Error: Too large voltage ({x_pos}V) provided for X (sine)!"
            )
            abort_move = True

        # CHECKING SINE INPUT VALUES TO SENT VIA DAC, ONLY POSITIVE VALUES ALLOWED
        if x_pos <= 0:
            self.logger_box.module_logger.info(
                f"Error: Negative voltage ({x_pos}V) provided for X (sweep), not allowed due to DAC!"
            )
            abort_move = True

        # ------------------------------------------

        # CHECKING COORDINATE VALUES
        if not (0 <= x_coord < self.step_dim):
            self.logger_box.module_logger.info(
                f"Aborted moving: Coordinates ({x_coord}, {y_coord}) must be in range ({0}, {self.step_dim-1}) "
            )
            abort_move = True

        if not (0 <= y_coord < self.step_dim):
            self.logger_box.module_logger.info(
                f"Aborted moving: Coordinates ({x_coord}, {y_coord}) must be in pixel range ({0}, {self.step_dim-1}) "
            )
            abort_move = True

        if not abort_move:
            self.logger_box.module_logger.info(
                f"Moving to pixels: ({x_coord}, {y_coord}) with voltages ({x_pos}, {y_pos})"
            )
            if not self.handle:
                self.open_labjack_connection()  # NOTE ONLINE ONLY
            ljm.eWriteNames(
                self.handle, 2, [self.x_address, self.y_address], [x_pos, y_pos]
            )
            self.logger_box.module_logger.info(f"Done moving!")
            # self.close_labjack_connection(closeserver=False)

        else:
            self.logger_box.module_logger.info(
                f"Aborted moving ({x_coord}, {y_coord}) with voltages ({x_pos}, {y_pos})"
            )

    # ----# NOTE FIX, THIS IS ONYL USED FOR DEMO MODE
    def temp_create_values_si(self):

        if self.prev_speed_mode == "slow":
            print("PREVIOUS SLOW MODE")
            # Up step vals: (Y)
            us_values = list(
                np.around(
                    np.linspace(
                        start=-self.step_amp,
                        stop=self.step_amp,
                        num=int(self.step_dim),
                        endpoint=True,
                    )
                    + self.step_offset,
                    decimals=10,
                )
            )
            # Sweep vals (X)
            sweep_values = np.around(
                np.linspace(
                    start=-self.sine_amp,
                    stop=self.sine_amp,
                    num=int(self.step_dim),
                    endpoint=True,
                )
                + self.sine_offset,
                decimals=10,
            )

        elif self.prev_speed_mode == "zoom":
            print("PREVIOUS ZOOM MODE")
            # Up step vals: (Y vals)
            us_values = list(
                np.around(
                    np.linspace(
                        start=self.minY,
                        stop=self.maxY,
                        num=int(self.step_dim),
                        endpoint=True,
                    ),
                    decimals=10,
                )
            )
            # Sweep vals X
            sweep_values = np.around(
                np.linspace(
                    start=self.minX,
                    stop=self.maxX,
                    num=int(self.step_dim),
                    endpoint=True,
                ),
                decimals=10,
            )
        else:
            self.abort_scan = True
            print("ERROR IN 'temp create values si()'")
            return None, None, None, None

        self.x_voltage_list = sweep_values.copy()
        self.y_voltage_list = us_values.copy()

    # -----

    def find_zoom_range(self, x_min, x_max, y_min, y_max):

        try:
            try:
                x_vals = self.x_voltage_list
                y_vals = self.y_voltage_list
                self.gui.logger_box.module_logger.info(f"FOUND X AND Y VOLTAGE LISTS!")

            except:
                self.gui.logger_box.module_logger.info(
                    f"FAILED TO FIND X AND Y VOLTAGE LISTS"
                )
                """
                self.abort_scan = False
                self.get_scan_parameters()
                self.temp_create_values_si()   # This to create the lists?"""
                print("abort scan=?", self.abort_scan)
                x_vals = self.x_voltage_list = self.gui.parent.t7.x_voltage_list
                y_vals = self.y_voltage_list = self.gui.parent.t7.y_voltage_list

            print(
                "x vals list:",
                x_vals[:5],
                "...",
                x_vals[-5:],
                "\n\ny vals list:",
                y_vals[:5],
                "...",
                y_vals[-5:],
            )
            """if self.speed_mode in ['fast', 'slow']:
                print("WARNING: FINDING ZOOM RANGE FROM FSAT OR SLOW RANGE")
                # RECREATING SCAN VALUES
                x_vals = np.around(np.linspace(start=-self.sine_amp, stop=self.sine_amp, num=int(self.step_dim), endpoint=True) + self.x_offset, decimals=10)
                y_vals = np.around(np.linspace(start=-self.step_amp, stop=self.step_amp, num=int(self.step_dim), endpoint=True) + self.y_offset, decimals=10)

            elif self.speed_mode == 'zoom':
                print("WARNING: FINDING ZOOM RANGE FROM ZOOM RANGE")
                x_vals = np.around(np.linspace(start=self.minX, stop=self.maxX, num=int(self.step_dim), endpoint=True), decimals=10)
                y_vals = np.around(np.linspace(start=self.minY, stop=self.maxY, num=int(self.step_dim), endpoint=True), decimals=10)"""

            # CHECKING IF OK
            if not (0 <= x_min < self.step_dim) or not (0 <= x_max < self.step_dim):
                self.logger_box.module_logger.info(
                    f"Aborted moving: X Coordinates ({x_min}, {x_max}) must be in range ({0}, {self.step_dim - 1}) "
                )
                return
            if not (0 <= y_min < self.step_dim) or not (0 <= y_max < self.step_dim):
                self.logger_box.module_logger.info(
                    f"Aborted moving: Y Coordinates ({y_min}, {y_max}) must be in pixel range ({0}, {self.step_dim - 1}) "
                )
                return

            # Extracting voltages for pixel range (NOTE THIS IS WITH OFFSET)
            x_pos_min = x_vals[x_min]
            x_pos_max = x_vals[x_max]
            y_pos_min = y_vals[y_min]
            y_pos_max = y_vals[y_max]

            self.gui.logger_box.module_logger.info(
                f"Chosen X voltages: ({x_pos_min}, {x_pos_max}) at pixels ({x_min}, {x_max})"
            )
            self.gui.logger_box.module_logger.info(
                f"Chosen Y voltages: ({y_pos_min}, {y_pos_max}) at pixels ({y_min}, {y_max})"
            )

        except:
            self.gui.logger_box.module_logger.info("Error getting move vars")
            raise
            return

        # TODO: ADD SAFETY CHECK BEFORE TRYING
        # ------------------------------------------
        abort_move = False

        if self.gui.demo_mode.get():
            # self.logger_box.module_logger.info("DEMO MODE: WILL NOT MOVE")
            abort_move = True

        # ----------- STEP BETWEEN ROWS (Y) -----------
        # CHECKING STEP INPUT VALUES TO SERVO, MAX ALLOWED IS 5V, WE HAVE 4V FOR MARGINS
        if abs(y_pos_min) > 4 or abs(y_pos_max) > 4:
            self.logger_box.module_logger.info(
                f"Error: Too large voltages ({y_pos_min}, {y_pos_max} V) provided for Y (step)!"
            )
            abort_move = True

        # ----------- STEP WITHIN ROW (X) -----------
        # CHECKING SINE INPUT VALUES TO SERVO, MAX ALLOWED IS 5V, WE HAVE 4V FOR MARGINS
        if abs(x_pos_min) > 4 or abs(x_pos_max) > 4:
            self.logger_box.module_logger.info(
                f"Error: Too large voltage ({x_pos_min}, {x_pos_max} V) provided for X (sine)!"
            )
            abort_move = True

        # CHECKING SINE INPUT VALUES TO SENT VIA DAC, ONLY POSITIVE VALUES ALLOWED
        if x_pos_min <= 0 or x_pos_max <= 0:
            self.logger_box.module_logger.info(
                f"Error: Negative voltage ({x_pos_min}, {x_pos_max} V) provided for X (sweep), not allowed due to DAC!"
            )
            abort_move = True

        # ------------------------------------------

        # CHECKING COORDINATE VALUES (Again????)

        if not (0 <= x_min < self.step_dim) or not (0 <= x_max < self.step_dim):
            self.logger_box.module_logger.info(
                f"Aborted moving: X Coordinates ({x_min}, {x_max}) must be in range ({0}, {self.step_dim - 1}) "
            )
            abort_move = True
        if not (0 <= y_min < self.step_dim) or not (0 <= y_max < self.step_dim):
            self.logger_box.module_logger.info(
                f"Aborted moving: Y Coordinates ({y_min}, {y_max}) must be in pixel range ({0}, {self.step_dim - 1}) "
            )
            abort_move = True

        # TODO: DO NEW SCAN HERE OR SET VARIABLES FOR NEW SCAN
        # .....
        # .....
        """ x_pos_min = x_vals[x_min]
            x_pos_max = x_vals[x_max]
            y_pos_min = y_vals[y_min]
            y_pos_max = y_vals[y_max]
            """
        if abort_move and not self.gui.demo_mode.get():
            print("ABORTED SETTING X Y MIN MAX")
            # x_pos_min = 0
            # x_pos_max = 0
            # y_pos_min = 0
            # y_pos_max = 0
        elif not abort_move or self.gui.demo_mode.get():
            self.gui.variable_containers["min X"]["var"].set(x_pos_min)
            self.gui.variable_containers["max X"]["var"].set(x_pos_max)
            self.gui.variable_containers["min Y"]["var"].set(y_pos_min)
            self.gui.variable_containers["max Y"]["var"].set(y_pos_max)
            self.gui.variable_containers["amp X"]["var"].set(
                round((x_pos_max - x_pos_min) / 2, 10)
            )
            self.gui.variable_containers["amp Y"]["var"].set(
                round((y_pos_max - y_pos_min) / 2, 10)
            )
            self.gui.variable_containers["scopelength"]["var"].set(
                self.gui.variable_containers["amp X"]["var"].get()
                * self.gui.variable_containers["lensslope"]["var"].get()
            )
        else:
            print("ELSE")
            input("waiting....")

    # MAIN FUNCTION THAT PREPARES AND PERFORMS SCAN:
    def fast_galvo_scan(self):
        self.abort_scan = False

        self.logger_box.module_logger.info(
            "Getting scan parameters and generating scan sweep and step values."
        )
        self.get_scan_parameters()
        self.get_step_values()
        self.get_sine_values(sweep_mode=self.gui.sweep_mode.get())

        self.logger_box.module_logger.info("Doing safety check on scan parameters.")
        self.safetytests.check_voltages()  # MOST IMPORTANT SO WE DON'T DAMAGE DEVICE WITH TOO HIGH VOLTAGE

        if not self.abort_scan:

            self.logger_box.module_logger.info("Opening labjack connection")
            if not self.offline:
                self.open_labjack_connection()  # NOTE ONLINE ONLY

                # err = ljm.eStreamStop(self.handle)   # UNCOMMENT IF WE GET STREAM ACTIVE ERROR (or maybe i included a button already)

            self.logger_box.module_logger.info("Populating command list")

            self.multi_populate_scan_cmd_list_burst()  # self.multi_populate_scan_lists()  #### NEW NAME???

            if not self.offline:
                self.fill_buffer_stream()  # NOTE ONLINE ONLY

            # Double check that scan command lists are safe
            self.logger_box.module_logger.info(
                "Checking for invalid addresses and values"
            )
            self.safetytests.multi_check_cmd_list(
                self.aAddressesUp, self.aValuesUp, check_txt="Up Check"
            )
            self.safetytests.multi_check_cmd_list(
                self.aAddressesDown, self.aValuesDown, check_txt="Down Check"
            )

            if self.abort_scan:
                self.logger_box.module_logger.info("Scan aborted")
                return

            self.logger_box.module_logger.info("Configuring Stream Trigger")
            if (
                self.useTrigger and not self.offline
            ):  # alternative is that we use "STREAM_ENABLE" as a sort of trigger

                if self.close_stream:
                    err = ljm.eStreamStop(
                        self.handle
                    )  # TODO: HANDLE ERROR IF STREAM IS ALREADY ACTIVE!
                    self.logger_box.module_logger.info(f"Close stream return: {err}")
                    self.close_stream = False

                self.configure_stream_trigger()  # NOTE ONLINE ONLY

            # Finish stream configs , replaces: ljm.eStreamStart(self.handle, self.b_scansPerRead,...)
            self.logger_box.module_logger.info("Configuring Stream Start")
            if not self.offline:
                self.configure_stream_start()

            self.logger_box.module_logger.info(
                "Creating socket connection with Qutag server."
            )
            if self.recordScan and not self.offline:
                self.socket_connection()

            # AGAIN FINAL CHECK, MAYBE REMOVE LATER
            self.logger_box.module_logger.info(
                "Final safety check on values and addresses"
            )
            self.gui.root.update()

            self.safetytests.multi_check_cmd_list(
                self.aAddressesUp, self.aValuesUp, check_txt="Up Check"
            )
            self.safetytests.multi_check_cmd_list(
                self.aAddressesDown, self.aValuesDown, check_txt="Down Check"
            )
            self.safetytests.check_voltages()  # MOST IMPORTANT SO WE DON'T DAMAGE DEVICE WITH TOO HIGH VOLTAGE
            # ----
            # self.abort_scan = True  # temp  # NOTE

            if not self.abort_scan:
                self.multi_start_scan()  # NOTE ONLINE ONLY
        else:
            self.logger_box.module_logger.info("Scan aborted after safety check.")

    def slow_galvo_scan(self, do_scan=True):

        def init_si():  # init slow improved
            self.abort_scan = False
            # self.int_time = self.gui.int_time.get()   # DWELL TIME FOR SLOW MODE
            self.get_scan_parameters()

        def connect_si():
            # TRY TO CREATE CONNECTION WITH LABJACK AND SERVER:
            try:
                if not self.offline:
                    self.logger_box.module_logger.info("Opening labjack connection")
                    self.open_labjack_connection()  # NOTE ONLINE ONLY

                if self.recordScan and not self.offline:
                    self.logger_box.module_logger.info(
                        "Creating socket connection with Qutag server."
                    )
                    self.socket_connection()
            except:
                print("FAILED CONNECTION (labjack or socket server)")
                self.close_labjack_connection()
                self.abort_scan = True

        def create_values_si():

            if self.speed_mode == "slow":
                # Up step vals: (Y)
                us_values = list(
                    np.around(
                        np.linspace(
                            start=-self.step_amp,
                            stop=self.step_amp,
                            num=int(self.step_dim),
                            endpoint=True,
                        )
                        + self.step_offset,
                        decimals=10,
                    )
                )
                # Sweep vals (X)
                sweep_values = np.around(
                    np.linspace(
                        start=-self.sine_amp,
                        stop=self.sine_amp,
                        num=int(self.step_dim),
                        endpoint=True,
                    )
                    + self.sine_offset,
                    decimals=10,
                )

            elif self.speed_mode == "zoom":
                # Up step vals: (Y vals)
                us_values = list(
                    np.around(
                        np.linspace(
                            start=self.minY,
                            stop=self.maxY,
                            num=int(self.step_dim),
                            endpoint=True,
                        ),
                        decimals=10,
                    )
                )
                # Sweep vals X
                sweep_values = np.around(
                    np.linspace(
                        start=self.minX,
                        stop=self.maxX,
                        num=int(self.step_dim),
                        endpoint=True,
                    ),
                    decimals=10,
                )
            else:
                self.abort_scan = True
                print("ERROR IN 'create values si()'")
                return None, None, None, None

            self.x_voltage_list = sweep_values.copy()
            self.y_voltage_list = us_values.copy()

            # Left and right sweep vals: (x lists)
            ls_values = list(sweep_values)
            rs_values = list(np.flip(sweep_values))
            return sweep_values, ls_values, rs_values, us_values

        def check_values_si():
            for sw_val in sweep_values_all:  # These must be between 0V and 5V
                if sw_val < 0.0:
                    self.abort_scan = True
                    print(f"Sweep voltage too low: {sw_val}")

                if sw_val > 4.0:
                    self.abort_scan = True
                    print(f"Sweep voltage too high: {sw_val}")

            for st_val in up_step_values:  # These must be between -5V and 5V
                if abs(st_val) > 4.0:
                    self.abort_scan = True
                    print(f"Step voltage too high: {st_val}")

            if len(add_list_01) != len(val_list_01):
                print("Error in length of 0.1s delay lists")
                self.abort_scan = True

            if len(add_list_001) != len(val_list_001):
                print("Error in length of 0.01s delay lists")
                self.abort_scan = True

            if len(add_list_dwell) != len(val_list_dwell):
                print("Error in length of dwell delay lists")
                self.abort_scan = True

        def plot_path_si(show=False, use_imshow=False):

            if use_imshow:
                fig, [ax1, ax2] = plt.subplots(1, 2, sharey=True, sharex=True)
                sweep_mat = np.zeros((self.step_dim, self.step_dim))
                step_mat = np.zeros((self.step_dim, self.step_dim))
                for s in range(len(up_step_values)):
                    if s % 2 == 0:
                        sweep_mat[s, :] = left_sweep_values
                    else:
                        sweep_mat[s, :] = right_sweep_values
                    step_mat[s, :] = np.ones(self.step_dim) * up_step_values[s]
                ax1.imshow(
                    step_mat, vmin=-0.3 + self.y_offset, vmax=0.3 + self.y_offset
                )
                ax2.imshow(
                    sweep_mat, vmin=-0.3 + self.x_offset, vmax=0.3 + +self.x_offset
                )

                ax1.set_title("Step matrix")
                ax2.set_title("Sweep matrix")
                return fig
            else:
                plt.figure()
                plt.plot(up_step_values, label="up step values")
                plt.plot(left_sweep_values, label="left sweep values")
                plt.plot(right_sweep_values, label="right sweep values")
                plt.legend()

            if show:
                plt.show()

        # Scanning using the Microcontroller
        def start_pos_si():
            if self.abort_scan:
                return
            self.logger_box.module_logger.info("Initializing start position")
            if not self.offline:  # ****
                ljm.eWriteNames(
                    self.handle,
                    2,
                    [self.step_addr, self.sine_addr],
                    [up_step_values[0], left_sweep_values[0]],
                )
                ljm.eWriteNames(
                    self.handle, len(add_list_001), add_list_001, val_list_001
                )  # sleep for 0.1s after step

            else:
                print(
                    f"DEMO: Would send start pos: {[self.step_addr, self.sine_addr]} --> {[up_step_values[0], left_sweep_values[0]]}"
                )

        def create_sweep_lists(dir):

            sweep_addresses_list_curr = []
            sweep_values_list_curr = []

            if dir == "left":
                sweep_vals_curr = left_sweep_values  # left sweep (every even row)
            elif dir == "right":
                sweep_vals_curr = right_sweep_values  # right sweep (every odd row)
            else:
                self.abort_scan = True
                return

            for sweep_idx in range(len(sweep_vals_curr)):
                # 1) Move to new pixel within row
                sweep_addresses_list_curr += [self.sine_addr]  # address of DAC
                sweep_values_list_curr += [sweep_vals_curr[sweep_idx]]

                # 1.5) Sleep after moving to wait for galvo for 0.01s
                sweep_addresses_list_curr += add_list_001
                sweep_values_list_curr += val_list_001

                # 2) Mark start of integration time
                if (
                    self.q_pingQuTag and self.ping102
                ):  # note: not using end sweep address
                    sweep_addresses_list_curr += [
                        self.q_M102_addr,
                        self.wait_address,
                        self.q_M102_addr,
                    ]
                    sweep_values_list_curr += [1, 1, 0]

                # 3) Wait integration time at pixel
                sweep_addresses_list_curr += add_list_dwell
                sweep_values_list_curr += val_list_dwell

                # 4) Mark end of integration time of pixel
                if self.q_pingQuTag and self.ping101:
                    sweep_addresses_list_curr += [
                        self.q_M101_addr,
                        self.wait_address,
                        self.q_M101_addr,
                    ]
                    sweep_values_list_curr += [1, 1, 0]

            return sweep_addresses_list_curr, sweep_values_list_curr

        def do_scan_si():
            if self.abort_scan:
                return
            # ---- DO SCAN -----
            self.logger_box.module_logger.info(f"Starting {self.speed_mode} scan")
            self.gui.root.update()
            time.sleep(3)
            start_time = time.time()

            # FOR EACH ROW
            for step_idx in range(len(up_step_values)):

                # ------- TODO: ------- requirement: threading to use the button and set cancel
                # if self.gui.state.get() == 'cancel':
                #    self.logger_box.module_logger.info(f"NOTE: SCAN WAS CANCELLED")
                #    self.gui.state.set('')
                #    break
                # ----------------------

                # TODO UPDATE PROGRESS BAR HERE
                # Progress bar update
                # proc_step += delta_proc
                # self.gui.pb['value'] += proc_step
                # self.gui.root.update()  # testing    # TODO NOTE FIXME, CHECK IF THIS AFFECTS ANYTHING TIME-WISE!!

                if not self.offline:  # ****
                    # 1) Take a step to new row!
                    res = ljm.eWriteNames(
                        self.handle, 1, [self.step_addr], [up_step_values[step_idx]]
                    )  # Take a step to new row
                    res = ljm.eWriteNames(
                        self.handle, len(add_list_001), add_list_001, val_list_001
                    )  # sleep for 0.01s after step

                    # 2) Do row sweep!
                    if step_idx % 2 == 0:  # left sweep (every even row)
                        res = ljm.eWriteNames(
                            self.handle,
                            len(sweep_addresses_cmds_left),
                            sweep_addresses_cmds_left,
                            sweep_values_cmds_left,
                        )
                    else:  # right sweep (every odd row)
                        res = ljm.eWriteNames(
                            self.handle,
                            len(sweep_addresses_cmds_right),
                            sweep_addresses_cmds_right,
                            sweep_values_cmds_right,
                        )

                """else:  # OFFLINE DEMO PRINT
                    print(f"({step_idx}) Step to new row: {[self.step_addr]} --> {[up_step_values[step_idx]]}")

                    if step_idx == 0:  # left sweep (every even row)
                        print(f"-- {step_idx}) Sweep new left sweep row:\n"
                          f"Add:{sweep_addresses_cmds_left[:5]}...{sweep_addresses_cmds_left[-5:]}\n"
                          f"Val:{sweep_values_cmds_left[:5]}...{sweep_values_cmds_left[-5:]}")
                    elif step_idx == 1:  # right sweep (every odd row)
                        print(f"-- {step_idx}) Sweep new right sweep row:\n "
                          f"   Addresses:{sweep_addresses_cmds_right[:5]}...{sweep_addresses_cmds_right[-5:]}\n"
                          f"   Values:{sweep_values_cmds_right[:5]}...{sweep_values_cmds_right[-5:]}")"""

            # ------
            end_time = time.time()
            self.logger_box.module_logger.info(
                f"Scan Done! (todo: check that file is complete before analysis)"
                f"\n   ETA scan time = {int(self.scanTime)} s"
                f"\n   Theoretical scan time = {self.step_dim * self.step_dim * self.int_time} s"  # note does not include nr of frames
                f"\n   Actual scan time   = {round(end_time - start_time, 6)} s"
            )
            self.gui.root.update()
            time.sleep(1)
            if not self.offline:  # ****
                # ----- reset galvo positions to offset:
                self.set_offset_pos()

                # ---- Tells server that we are done scanning
                print("Stopping server")

                self.socket_connection(doneScan=True)
                # time.sleep(1)

        def add_wait_delay_si(delay_s):
            add_list = []
            val_list = []
            curr_delay = 0

            s_to_um = 1000000  # 1e6
            delay_max = 100000  # 1e5 -->  max to send to wait address is 0.1 seconds

            # print(f"\nTOTAL DELAY TO ADD = {delay_s} s ---> {delay_s*s_to_um} us")
            # print(f"NR OF FULL DELAYS = {delay_s//0.1}")

            # Add as many full 0.1s delays as we can fit in total delay
            for i in range(int(delay_s / 0.1)):
                add_list += [self.wait_address]
                val_list += [
                    delay_max
                ]  # note: delay_max is defined to be a full 0.1s which is max for the wait address
                curr_delay += delay_max

            # Add any residual delay (seconds)
            res_delay_us = int(
                (delay_s * s_to_um) - curr_delay
            )  # we do this to avoid rounding errors in subtraction of small numbers

            if res_delay_us < 0:
                print("TOO MUCH DELAY ADDED!!!")
                self.abort_scan = True

            elif res_delay_us == 0:
                print("EXACT DELAY ADDED, NO RESIDUAL")

            elif 0 < res_delay_us < delay_max:
                # res_delay_us = (round(self.step_delay / 0.1, 10) - int(self.step_delay / 0.1)) * 0.1 * 1000000
                print(
                    f"REMAINING DELAY TO ADD = {res_delay_us/s_to_um} s ---> {res_delay_us} us"
                )
                add_list += [self.wait_address]
                val_list += [res_delay_us]

            else:
                self.abort_scan = True
                print("Error in delay function")

            return add_list, val_list

        def check_cmds(addresses, values):
            # self.logger_box.module_logger.info(check_txt)

            # 1) WE CHECK THE COMMAND LIST LENGTHS
            if len(addresses) != len(values):
                self.logger_box.module_logger.info(
                    "ERROR. NOT SAME COMMAND LIST LENGTHS. MISALIGNMENT DANGER."
                )
                self.abort_scan = True

            # 2) WE CHECK THE ADDRESSES IN CMD LIST
            for i in range(len(addresses)):
                # CHECK WAIT CMDS
                if addresses[i] == self.wait_address:
                    if not (0 < values[i] <= 100000):
                        self.logger_box.module_logger.info(
                            "ERROR. ", values[i], " WAIT VALUE IS NOT IN RANGE"
                        )
                        self.abort_scan = True
                # CHECK STEP CMDS
                elif (
                    addresses[i] == self.step_addr
                ):  # currently step is outside cmd list
                    self.logger_box.module_logger.info(
                        "ERROR. STEP VALUE IN COMMAND LIST"
                    )
                    self.abort_scan = True
                    if abs(values[i]) > 4:
                        self.logger_box.module_logger.info("ERROR. STEP VALUE TOO BIG")
                        self.abort_scan = True
                # CHECK SWEEP CMDS
                elif addresses[i] == self.sine_addr:
                    if not (0 < values[i] < 4):
                        self.logger_box.module_logger.info(
                            "ERROR. SWEEP VALUE NOT IN RANGE (0, 4) V"
                        )
                        self.abort_scan = True
                # CHECKING MARKER CMDS
                elif (addresses[i] == self.q_M101_addr) or (
                    addresses[i] == self.q_M102_addr
                ):
                    if values[i] != 0 and values[i] != 1:
                        self.logger_box.module_logger.info(
                            "ERROR. MARKER VALUE ERROR. MUST BE IN {0,1}"
                        )
                        self.abort_scan = True
                else:
                    self.logger_box.module_logger.info(
                        f"'{addresses[i]}' ... Address not recognized or checked for in 'check_cmd_list()'. Aborting scan."
                    )
                    self.abort_scan = True

            if self.abort_scan:
                self.logger_box.module_logger.info("Final Check Failed...\n")
            else:
                pass
                # self.logger_box.module_logger.info("Final Check Succeeded!\n")

        # proc_step = 0   # FOR PROGRESS BAR
        # delta_proc = int(100//(self.step_dim*self.step_dim))
        # ----------------------------

        print("IN IMPROVED SLOW MODE")
        # 1) Initialize scan params
        init_si()
        # 2) Connect to LJ and server
        if do_scan:
            connect_si()
        # 3)
        try:
            # CREATE DATA TO USE
            sweep_values_all, left_sweep_values, right_sweep_values, up_step_values = (
                create_values_si()
            )

            add_list_01, val_list_01 = add_wait_delay_si(
                0.1
            )  # addresses and values for 0.1s delay
            add_list_001, val_list_001 = add_wait_delay_si(
                0.001
            )  # addresses and values for 0.01s delay waiting for galvo to move to pixel
            add_list_dwell, val_list_dwell = add_wait_delay_si(
                self.gui.int_time.get()
            )  # addresses and values for dwell time delay

            sweep_addresses_cmds_left, sweep_values_cmds_left = create_sweep_lists(
                "left"
            )
            sweep_addresses_cmds_right, sweep_values_cmds_right = create_sweep_lists(
                "right"
            )

            # CHECK VALUES TO SEND IN
            check_values_si()  # TODO: CHECK SWEEP CMDS
            check_cmds(sweep_addresses_cmds_left, sweep_values_cmds_left)
            check_cmds(sweep_addresses_cmds_right, sweep_values_cmds_right)

            # PLOT VALUES TO SEND IN
            if False:
                plot_path_si(show=True)

            # FINAL CHECK TO ABORT
            if self.abort_scan:  # or self.offline:  # last line of defense
                self.logger_box.module_logger.info("Aborted scan due to error")
                return False  # note: comment this out if we want to check the prints/simulation

            if not do_scan and self.offline:  # or self.offline:  # last line of defense
                return plot_path_si(show=True, use_imshow=True)
                # return False

            if do_scan:
                # INITALIZING START POSITIONS
                start_pos_si()

                # DO SCAN
                do_scan_si()

            return True

        except ljm.LJMError:
            if not self.offline:
                self.logger_box.module_logger.info("Failed scan")
                self.close_labjack_connection()
                raise
            return False

    # Step 1) Sets all parameters depending on selected scan pattern and scan type
    def get_scan_parameters(self):
        # --------------- HARDCODED FOR THIS SIMPLER METHOD ------------------------------
        self.sine_addr = self.x_address
        self.sine_offset = self.x_offset
        self.step_addr = self.y_address
        self.step_offset = self.y_offset
        # --------------- Chosen scan parameters ----------------------------------------

        self.speed_mode = self.gui.speed_mode.get()  # 'fast' or 'slow'
        self.filename = self.gui.data_file.get()  # self.scanVariables.filename
        self.num_frames = (
            self.gui.nr_frames.get()
        )  # self.scanVariables.num_frames  # NOTE: NEW PARAM # how many frames/images we want to scan
        self.step_amp = (
            self.gui.ampY.get()
        )  # self.scanVariables.step_voltage  # voltage = angle*0.22
        self.step_dim = self.gui.dimY.get()  # self.scanVariables.step_dim

        self.recordScan = self.gui.record_mode.get()  # self.scanVariables.recordScan
        self.diagnostics = (
            self.gui.diagnostics_mode.get()
        )  # self.scanVariables.diagnostics
        self.offline = self.gui.demo_mode.get()  # self.scanVariables.offline

        self.q_pingQuTag = True  # self.scanVariables.pingQuTag   # default = True
        self.useTrigger = True  # self.scanVariables.useTrigger  # default = True
        self.ping101 = True  # self.scanVariables.ping101  # marker before step  # default = True --> marker AFTER  step, after sweep ends
        self.ping102 = True  # self.scanVariables.ping102  # marker after step   # default = True --> marker BEFORE step, before sweep starts

        # self.data_folder =... '/Users/juliawollter/Desktop/Microscope GUI/Data')  # TODO FIXME: unused gui parameter (for now)

        # -------

        # --------------- PLACEHOLDER VALUES --------------------------------------------
        # List of x and y values, and lists sent to Labjack:
        self.step_values = []  # values to step through
        self.step_values_up = []  # NOTE: NEW VARIABLE
        self.step_values_down = []  # NOTE: NEW VARIABLE

        self.step_times = []  # for plotting (single period)
        self.sine_values = []  # values for one sine period, for buffer
        self.sine_times = []  # for plotting (single period)
        # self.aAddresses = []
        # self.aValues = []
        self.aAddressesUp = []  # NOTE: NEW PARAM
        self.aValuesUp = []  # NOTE: NEW PARAM
        self.aAddressesDown = []  # NOTE: NEW PARAM
        self.aValuesDown = []  # NOTE: NEW PARAM
        # ------- ZOOM

        self.minX = self.gui.minX.get()
        self.maxX = self.gui.maxX.get()
        self.minY = self.gui.minY.get()
        self.maxY = self.gui.maxY.get()

        # --------------- SINE ------------------------------
        self.b_max_buffer_size = 512  # Buffer stream size for y waveform values. --> Becomes resolution of sinewave period waveform == y_steps . i think it is max 512 samples (16-bit samples)?
        # Sine waveform:
        self.sine_amp = self.gui.ampX.get()  # self.scanVariables.sine_voltage
        self.sine_freq = self.gui.freq.get()  # self.scanVariables.sine_freq
        self.sine_period = 1 / self.sine_freq
        self.sine_phase = np.pi / 2
        self.sine_dim = int(
            self.b_max_buffer_size / 2
        )  # sine_dim = samplesToWrite = how many values we save to buffer stream = y_steps = resolution of one period of sinewave, --> sent to TickDAC --> sent to y servo input
        self.sine_delay = (
            self.sine_period / self.sine_dim
        )  # time between each y value in stream buffer     #self.sine_delay = 1 / (self.sine_dim / (2 * self.step_delay))
        if self.speed_mode == "fast":
            # Buffer stream variables:
            self.b_scanRate = int(
                self.sine_dim / self.sine_period
            )  # scanrate = scans per second = samples per second for one address = (resolution for one sine period)/(one sine period)   NOTE: (2*self.step_delay) = self.sine_period (of sinewave)
            # TODO: what happens if we set "b_scansPerRead" to 0 instead?
            self.b_scansPerRead = (
                self.b_scanRate
            )  # int(self.b_scanRate / 2)  # NOTE: When performing stream OUT with no stream IN, ScansPerRead input parameter to LJM_eStreamStart is ignored. https://labjack.com/pages/support/?doc=%2Fsoftware-driver%2Fljm-users-guide%2Festreamstart
            self.b_targetAddress = ljm.nameToAddress(self.sine_addr)[0]
            self.b_streamOutIndex = 0  # index of: "STREAM_OUT0" I think this says which stream you want to get from (if you have several)
            self.b_aScanList = [
                ljm.nameToAddress("STREAM_OUT0")[0]
            ]  # "STREAM_OUT0" == 4800
            self.b_nrAddresses = 1
        # -----------------------
        self.extra_delay = 0.001  # extra delay (seconds) to ensure that sine curve has reached a minimum/end
        self.step_delay = (
            self.sine_period + self.extra_delay
        )  # time between every X command. Should be half a period (i.e. time for one up sweep)
        print("period", self.sine_period, "step delay", self.step_delay, "+", 0.001)
        # calculates constants we need to do wait_us_blocking for any frequency. NOTE!!! Can be moved to get_params func
        # Calculate residual delay for step delay (a full period)
        self.wait_delay = (
            0.1 * 1000000
        )  # wait_delay = self.step_delay * 1000000   # "Delays for x microseconds. Range is 0-100000
        coveredDelay = 0.1 * int(self.step_delay / 0.1)
        self.remaining_delay = (
            (round(self.step_delay / 0.1, 10) - int(self.step_delay / 0.1))
            * 0.1
            * 1000000
        )

        # self.logger_box.module_logger.info("total delay:", round(self.step_delay, 6))
        # self.logger_box.module_logger.info("covered delay:", round(coveredDelay, 6), "seconds")
        # self.logger_box.module_logger.info("remaining delay:", round(self.step_delay - coveredDelay, 6), "?=", self.remaining_delay/1000000)
        # -----------------------
        # Expected scan time:

        if self.speed_mode == "fast":
            self.scanTime = (
                self.num_frames * 1.1 * self.step_dim * self.step_delay
            )  # Expected time sent to qutag server    Note: it will be slightly higher than this which depends on how fast labjack can iterate between commands
            self.scanTime += 10  # FIXME

        else:
            self.scanTime = 10 + (
                self.num_frames
                * self.step_dim
                * self.step_dim
                * (self.gui.int_time.get() + 0.001)
            )
            self.int_time = self.gui.int_time.get()

    def get_scan_parameters_copy(self, parent):
        self.prev_speed_mode = copy.deepcopy(parent.speed_mode)

        # --------------- HARDCODED FOR THIS SIMPLER METHOD ------------------------------
        self.sine_addr = copy.deepcopy(parent.sine_addr)
        self.sine_offset = copy.deepcopy(parent.sine_offset)
        self.step_addr = copy.deepcopy(parent.step_addr)
        self.step_offset = copy.deepcopy(parent.step_offset)
        # --------------- Chosen scan parameters --
        self.speed_mode = copy.deepcopy(parent.speed_mode)
        self.filename = copy.deepcopy(parent.filename)
        self.num_frames = copy.deepcopy(parent.num_frames)
        self.step_amp = copy.deepcopy(parent.step_amp)
        self.step_dim = copy.deepcopy(parent.step_dim)
        self.recordScan = copy.deepcopy(parent.recordScan)
        self.diagnostics = copy.deepcopy(parent.diagnostics)
        self.offline = copy.deepcopy(parent.offline)
        self.q_pingQuTag = copy.deepcopy(parent.q_pingQuTag)
        self.useTrigger = copy.deepcopy(parent.useTrigger)
        self.ping101 = copy.deepcopy(parent.ping101)
        self.ping102 = copy.deepcopy(parent.ping102)
        # --------------- PLACEHOLDER VALUES ------
        # List of x and y values, and lists sent to
        self.step_values = copy.deepcopy(parent.step_values)
        self.step_values_up = copy.deepcopy(parent.step_values_up)
        self.step_values_down = copy.deepcopy(parent.step_values_down)
        self.step_times = copy.deepcopy(parent.step_times)
        self.sine_values = copy.deepcopy(parent.sine_values)
        self.sine_times = copy.deepcopy(parent.sine_times)
        self.aAddressesUp = copy.deepcopy(parent.aAddressesUp)
        self.aValuesUp = copy.deepcopy(parent.aValuesUp)
        self.aAddressesDown = copy.deepcopy(parent.aAddressesDown)
        self.aValuesDown = copy.deepcopy(parent.aValuesDown)

        # ------- ZOOM
        self.minX = copy.deepcopy(parent.minX)
        self.maxX = copy.deepcopy(parent.maxX)
        self.minY = copy.deepcopy(parent.minY)
        self.maxY = copy.deepcopy(parent.maxY)

        self.x_voltage_list = copy.deepcopy(parent.x_voltage_list)
        self.y_voltage_list = copy.deepcopy(parent.y_voltage_list)

        # --------------- SINE --------------------
        self.b_max_buffer_size = parent.b_max_buffer_size
        # Sine waveform:
        self.sine_amp = copy.deepcopy(parent.sine_amp)
        self.sine_freq = copy.deepcopy(parent.sine_freq)
        self.sine_period = copy.deepcopy(parent.sine_period)
        self.sine_phase = copy.deepcopy(parent.sine_phase)
        self.sine_dim = copy.deepcopy(parent.sine_dim)
        self.sine_delay = copy.deepcopy(parent.sine_delay)

        # TODO FIX
        """if copy.deepcopy(parent.speed_mode == 'fast':
            # Buffer stream variables:
            self.b_scanRate =       copy.deepcopy(parent.b_scanRate      )
            self.b_scansPerRead =   copy.deepcopy(parent.b_scansPerRead  )
            self.b_targetAddress =  copy.deepcopy(parent.b_targetAddress )
            self.b_streamOutIndex = copy.deepcopy(parent.b_streamOutIndex)
            self.b_aScanList =      copy.deepcopy(parent.b_aScanList     )
            self.b_nrAddresses =    copy.deepcopy(parent.b_nrAddresses"""
        # -----------------------
        self.extra_delay = copy.deepcopy(parent.extra_delay)
        self.step_delay = copy.deepcopy(parent.step_delay)
        self.wait_delay = copy.deepcopy(parent.wait_delay)
        self.remaining_delay = copy.deepcopy(parent.remaining_delay)

        self.scanTime = copy.deepcopy(parent.scanTime)
        self.int_time = copy.deepcopy(parent.int_time)

    # Step 2) Returns a list of step and sine values that the scan will perform

    def get_step_values(self):
        # populating "step_values" list with discrete values
        step_size = (2 * self.step_amp) / (
            self.step_dim - 1
        )  # step size of our x values
        k = -self.step_amp
        for i in range(self.step_dim):
            self.step_times.append(i * self.step_delay)  # for plotting
            self.step_values.append(round(k + self.step_offset, 10))
            k += step_size

        self.y_voltage_list = self.step_values.copy()  # NEW BUT DON*T TRUST YET

    def get_sine_values(self, sweep_mode="linear"):  # sine waveform
        # Change compared to before: now we don't ensure exactly symmetrical sine values for up/down sweeps.
        self.sine_times = list(
            np.arange(start=0, stop=self.sine_dim, step=1) * self.sine_delay
        )  # for plotting if desired

        self.logger_box.module_logger.info(
            f"Using sweep mode '{sweep_mode}' for buffer values"
        )

        if sweep_mode == "linear":
            half_lin_values = np.around(
                np.linspace(
                    start=-self.sine_amp,
                    stop=self.sine_amp,
                    num=int(self.sine_dim / 2),
                    endpoint=True,
                )
                + self.sine_offset,
                decimals=10,
            )
            self.sine_values = list(half_lin_values) + list(np.flip(half_lin_values))
            if self.sine_dim != len(self.sine_values):
                self.logger_box.module_logger.info(
                    "Error: length of sweep list must be an even amount"
                )
                self.abort_scan = True

            self.x_voltage_list = (
                half_lin_values.copy()
            )  # NOTE NOT SURE IF WORKS WITH MOVE TO POS BUT ANYWAY FAST MODE NEEDS WORK

        # NOTE: WE ARE NOT DOING SINE ANYMORE, WILL REMOVE LATER!
        elif sweep_mode == "sine":
            print("ERROR: Sine method is not used anymore!!")
            self.logger_box.module_logger.info(
                f"ERROR: Sine method is not used anymore!!"
            )
            # sine_fast = self.sine_amp * np.sin((2 * np.pi * self.sine_freq * np.array(self.sine_times)) - self.sine_phase) + self.sine_offset
            # self.sine_values = list(np.around(sine_fast, decimals=10))
            self.abort_scan = True

        else:
            print("Unknown sweep mode")
            self.logger_box.module_logger.info(
                f"ERROR: Unknown sweep mode for buffer list"
            )
            self.abort_scan = True
            return

    # Step 4) Connect to LabJack device
    def open_labjack_connection(self):
        self.handle = ljm.openS(
            "T7", "ANY", "ANY"
        )  # ErrorCheck(self.handle, "LJM_Open")
        info = ljm.getHandleInfo(
            self.handle
        )  # ErrorCheck(info, "PrintDeviceInfoFromHandle")
        # self.logger_box.module_logger.info(f"Opened a LabJack with Device type: {info[0]}, Connection type: {info[1]},\n "
        # f"Serial number: {info[2]}, IP address: {ljm.numberToIP(info[3])}, Port: {info[4]},\n"
        # f"Max bytes per MB: {info[5]} \n")
        self.logger_box.module_logger.info(
            f"Opened a LabJack with Device type: {info[0]}, Connection type: {info[1]},\n"
            f"Serial number: {info[2]}, IP address: {ljm.numberToIP(info[3])}, Port: {info[4]},\n"
            f"Max bytes per MB: {info[5]} \n"
        )

    # Step 5) Connect to qu_tag
    def socket_connection(self, shutdown_server=False, doneScan=False):
        """Sets up a server ot communciate to the qutag computer to start a measurement
        Sends the file and scan time to the computer"""

        if self.diagnostics:
            print("Selected diagnostic mode, skipping connection to SSPD computer")
            return

        if self.gui.demo_mode.get():
            print("DEMO MODE, DID NOT SOCKET CONNECT")
            return  # in demo we don't want to create a server

        if shutdown_server:
            printlog = False  # will not log it but print instead
        else:
            printlog = True

        HEADERSIZE = 10
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Establishes a server
        # host = socket.gethostname()
        host = "130.237.35.177"  # IP address of this computer
        s.bind((host, 55555))
        s.listen(5)  # (10)

        self.print_log(f"Setting up the server at: {host}", printlog)
        run_flag = True
        while run_flag:  # Keep looking for a connection
            clientsocket, address = s.accept()
            self.print_log(f"Connection from {address} has been established!", printlog)

            # Establish that a connection has been made and sends a greeting
            msg = "welcome to the server!"
            msg = pickle.dumps(msg)
            msg = bytes(f"{len(msg):<{HEADERSIZE}}", "utf-8") + msg
            r1 = clientsocket.send(msg)

            # Sends the relevant information
            # Mode is the qutag mode to produce a txt(0) or timeres file (1)
            if shutdown_server:
                mode = 7  # this indicates to the ssdp side that we are done
                self.print_log(f"Sending shutdown code!", printlog)
            elif doneScan:
                mode = 0  # this indicates that our scan is done?
            else:
                mode = 1

            msg = {"file": self.filename, "scantime": self.scanTime, "mode": mode}
            msg = pickle.dumps(msg)
            msg = bytes(f"{len(msg):<{HEADERSIZE}}", "utf-8") + msg
            r2 = clientsocket.send(msg)
            if clientsocket:
                print(
                    "----- WAITING 5 SECONDS AFTER CONNECTION, CHECK IF WE CAN REMOVE THIS -----"
                )
                time.sleep(5)
                print("---- done waiting ----")
                # windowgui.root.after(int(2 * 1000), windowgui.sleep)  # time.sleep(5)    # Give the qutag a few seconds to start up
                break

    def multi_add_to_up_command_lists(self, addresses, values):
        # This is a precaution to prevent adding to one list without the other
        self.aAddressesUp += addresses
        self.aValuesUp += values

    def multi_add_to_down_command_lists(self, addresses, values):
        # This is a precaution to prevent adding to one list without the other
        self.aAddressesDown += addresses
        self.aValuesDown += values

    # Step 6) Adds x values and qtag pings and other commands to command list
    def multi_populate_scan_cmd_list_burst(
        self,
    ):  # USE TRIGGER WE HAVE SET UP PREVIOUSLY

        self.step_values_up = self.step_values.copy()  # NOTE: NEW ADDITIONS
        self.step_values_down = self.step_values.copy()  # NOTE: NEW ADDITIONS
        self.step_values_down.reverse()  # NOTE: NEW ADDITIONS

        self.cmd_pulse_trigger(state="arm")
        for step_idx in range(len(self.step_values)):
            self.cmd_marker(102)  # marker ch4

            self.cmd_step_value(step_idx)  # take step

            self.cmd_marker(101)  # marker ch4

            self.cmd_pulse_trigger(state="fire")  # send value 0

            self.multi_add_wait_delay()  # waits a whole period and a delta extra
            # ???? do below instead of multi_add_wait_delay to see that we do need to wait a full period
            # self.aAddresses += [self.wait_address]
            # self.aValues += [self.wait_delay]

            # RESETTING TRIGGER ETC:
            self.cmd_enable_trigger("off")
            self.cmd_pulse_trigger(state="arm")
            self.reset_num_scans()  # NEED TO RESET STUFF
            self.cmd_enable_trigger("on")

    def test_multi_populate_scan_cmd_list_burst(
        self,
    ):  # USE TRIGGER WE HAVE SET UP PREVIOUSLY
        # self.logger_box.module_logger.info("OPTION 1: external trigger")
        """
        _____________________________________________

        PREV METHOD:
        > trigger stream
        > for i in range(dimX):
            > marker 101 (maybe)
            > step
            > marker 102
            > wait --> t=period
        _____________________________________________

        NEW METHOD:
        arm trigger
        > repeat:
            > step
            > marker 101
            > fire trigger
            > wait --> t=period+delta
            > marker 102 (maybe)  ...  or this should be before we step?
            > reset trigger and stream configs for next round
        _____________________________________________
        """
        # do below instead of multi_add_wait_delay to see that we do need to wait a full period
        # self.aAddresses += [self.wait_address]
        # self.aValues += [self.wait_delay]

        self.step_values_up = self.step_values.copy()  # NOTE: NEW ADDITIONS
        self.step_values_down = self.step_values.copy()  # NOTE: NEW ADDITIONS
        self.step_values_down.reverse()  # NOTE: NEW ADDITIONS

        self.cmd_pulse_trigger(state="arm")
        for step_idx in range(len(self.step_values)):
            self.cmd_marker(102)
            self.cmd_step_value(step_idx)
            self.cmd_marker(101)

            self.cmd_pulse_trigger(state="fire")
            self.multi_add_wait_delay()  # waits a period and a delta extra
            # RESETTING TRIGGER ETC:
            self.cmd_enable_trigger("off")
            self.cmd_pulse_trigger(state="arm")
            self.reset_num_scans()  # NEED TO RESET STUFF
            self.cmd_enable_trigger("on")

            """
            self.cmd_pulse_trigger(state="fire")
            self.multi_add_wait_delay()  # waits a period and a delta extra
            # RESETTING TRIGGER ETC:
            self.cmd_enable_trigger("off")
            self.cmd_pulse_trigger(state="arm")
            self.reset_num_scans()  # NEED TO RESET STUFF
            self.cmd_enable_trigger("on")"""

    def reset_num_scans(self):
        # self.aAddresses += ["STREAM_NUM_SCANS"]; self.aValues += [self.sine_dim]  # [int(self.sine_dim/2)]  # [self.sine_dim]
        self.multi_add_to_up_command_lists(
            addresses=["STREAM_NUM_SCANS"], values=[self.sine_dim]
        )  # NOTE: NEW ADDITIONS
        self.multi_add_to_down_command_lists(
            addresses=["STREAM_NUM_SCANS"], values=[self.sine_dim]
        )  # NOTE: NEW ADDITIONS

    def multi_add_wait_delay(self):
        # Add as many 0.1s delays as we can fit
        for i in range(int(self.step_delay / 0.1)):
            # self.aAddresses += [self.wait_address] ; self.aValues += [self.wait_delay]
            self.multi_add_to_up_command_lists(
                addresses=[self.wait_address], values=[self.wait_delay]
            )  # NOTE: NEW ADDITIONS
            self.multi_add_to_down_command_lists(
                addresses=[self.wait_address], values=[self.wait_delay]
            )  # NOTE: NEW ADDITIONS

        # Add any residual delay
        if self.remaining_delay > 0:
            # self.aAddresses += [self.wait_address] ; self.aValues += [self.remaining_delay]
            self.multi_add_to_up_command_lists(
                addresses=[self.wait_address], values=[self.remaining_delay]
            )  # NOTE: NEW ADDITIONS
            self.multi_add_to_down_command_lists(
                addresses=[self.wait_address], values=[self.remaining_delay]
            )  # NOTE: NEW ADDITIONS

    def cmd_marker(self, marker):
        # Add "step marker"
        if self.q_pingQuTag:
            if marker == 101 and self.ping101:
                # self.aAddresses += [self.q_M101_addr, self.q_M101_addr]; self.aValues += [1, 0]
                self.multi_add_to_up_command_lists(
                    addresses=[self.q_M101_addr, self.wait_address, self.q_M101_addr],
                    values=[1, 1, 0],
                )  # NOTE: NEW ADDITIONS
                self.multi_add_to_down_command_lists(
                    addresses=[self.q_M101_addr, self.wait_address, self.q_M101_addr],
                    values=[1, 1, 0],
                )  # NOTE: NEW ADDITIONS

            elif marker == 102 and self.ping102:  # note: not using end sweep address
                # self.aAddresses += [self.q_M102_addr, self.q_M102_addr]; self.aValues += [1, 0]
                self.multi_add_to_up_command_lists(
                    addresses=[self.q_M102_addr, self.wait_address, self.q_M102_addr],
                    values=[1, 1, 0],
                )  # NOTE: NEW ADDITIONS
                self.multi_add_to_down_command_lists(
                    addresses=[self.q_M102_addr, self.wait_address, self.q_M102_addr],
                    values=[1, 1, 0],
                )  # NOTE: NEW ADDITIONS
            else:
                pass

    def cmd_pulse_trigger(self, state):
        if self.useTrigger:
            # Send a falling edge to the source of the trigger pulse, which is connected to the trigger channel --> Triggers stream.
            if state == "arm":
                # self.aAddresses += [self.tr_source_addr]; self.aValues += [1]     # arm/setup trigger --> 1=High
                self.multi_add_to_up_command_lists(
                    addresses=[self.tr_source_addr], values=[1]
                )  # NOTE: NEW ADDITIONS
                self.multi_add_to_down_command_lists(
                    addresses=[self.tr_source_addr], values=[1]
                )  # NOTE: NEW ADDITIONS
            elif (
                state == "fire"
            ):  # trigger is set off by falling edge (edge from 1 to 0)
                # self.aAddresses += [self.tr_source_addr]; self.aValues += [0]     # execute trigger --> 0=Low
                self.multi_add_to_up_command_lists(
                    addresses=[self.tr_source_addr], values=[0]
                )  # NOTE: NEW ADDITIONS
                self.multi_add_to_down_command_lists(
                    addresses=[self.tr_source_addr], values=[0]
                )  # NOTE: NEW ADDITIONS
        else:
            self.logger_box.module_logger.info(
                "Error. Incorrect trigger based on 'useTrigger' parameter."
            )

    def cmd_enable_trigger(self, state):
        # instead of jumper trigger, use "ENABLE_STREAM"
        if self.useTrigger:  # if not self.useTrigger: before
            if state == "on":
                # self.aAddresses += ["STREAM_ENABLE"] ; self.aValues += [1]  # 1=High
                self.multi_add_to_up_command_lists(
                    addresses=["STREAM_ENABLE"], values=[1]
                )  # NOTE: NEW ADDITIONS
                self.multi_add_to_down_command_lists(
                    addresses=["STREAM_ENABLE"], values=[1]
                )  # NOTE: NEW ADDITIONS

            elif state == "off":
                # self.aAddresses += ["STREAM_ENABLE"] ; self.aValues += [0]  # 0=Low
                self.multi_add_to_up_command_lists(
                    addresses=["STREAM_ENABLE"], values=[0]
                )  # NOTE: NEW ADDITIONS
                self.multi_add_to_down_command_lists(
                    addresses=["STREAM_ENABLE"], values=[0]
                )  # NOTE: NEW ADDITIONS
            else:
                self.logger_box.module_logger.info("Error in enable stream")
                self.abort_scan = True
        else:
            self.logger_box.module_logger.info(
                "Error. Incorrect enable trigger based on 'useTrigger' parameter."
            )

    def cmd_step_value(self, idx):
        # Add step value
        # self.aAddresses += [self.step_addr] ; self.aValues += [step]
        self.multi_add_to_up_command_lists(
            addresses=[self.step_addr], values=[self.step_values_up[idx]]
        )  # NOTE: NEW ADDITIONS
        self.multi_add_to_down_command_lists(
            addresses=[self.step_addr], values=[self.step_values_down[idx]]
        )  # NOTE: NEW ADDITIONS

    # Step 7) Write sine waveform values to stream buffer (memory)
    def fill_buffer_stream(self):
        # https://labjack.com/pages/support?doc=/datasheets/t-series-datasheet/32-stream-mode-t-series-datasheet/#section-header-two-ttmre
        try:
            # self.logger_box.module_logger.info("Initializing stream out... \n")
            err = ljm.periodicStreamOut(
                self.handle,
                self.b_streamOutIndex,
                self.b_targetAddress,
                self.b_scanRate,
                self.sine_dim,
                self.sine_values,
            )
            # self.logger_box.module_logger.info("Write to buffer error =", err)
        except ljm.LJMError:
            self.logger_box.module_logger.info("Failed upload buffer vals")
            # ljm_stream_util.prepareForExit(self.handle)
            self.close_labjack_connection()
            raise

    def configure_stream_start(self):
        # previously --> ljm.eStreamStart(self.handle, self.b_scansPerRead, self.b_nrAddresses, self.b_aScanList, self.b_scanRate)
        try:
            # self.b_scansPerRead   TODO check
            # self.b_nrAddresses    done
            # self.b_aScanList      done
            # self.b_scanRate)      TODO check
            # NUM SCANS WORKS WITH PERIODIC SETUP
            # TODO: change back below
            ljm.eWriteName(
                self.handle, "STREAM_NUM_SCANS", self.sine_dim
            )  # int(self.sine_dim/2))  # = 256, how many values in buffer we want to burst stream (full period of values)
            ljm.eWriteName(self.handle, "STREAM_SCANRATE_HZ", self.b_scanRate)  #
            ljm.eWriteName(
                self.handle, "STREAM_NUM_ADDRESSES", self.b_nrAddresses
            )  # len(b_aScanList), nr of output channels/streams
            # ljm.eWriteName(self.handle, "STREAM_AUTO_TARGET", )                                       # TODO CHECK IF NEEDED
            ljm.eWriteName(
                self.handle, "STREAM_SCANLIST_ADDRESS0", self.b_aScanList[0]
            )  # TODO CHECK IF NEEDED AND WHAT IT IS
            # ljm.eWriteName(self.handle, "STREAM_DATATYPE", 0)                                         # ???? TODO CHECK IF NEEDED
            if self.useTrigger:
                ljm.eWriteName(
                    self.handle, "STREAM_ENABLE", 1
                )  # ???? TODO CHECK IF NEEDED
            # TODO: READ BACK ACTUAL SCAN RATE
            # self.logger_box.module_logger.info("Scan Rate:", self.b_scanRate, "vs.", scanRate)
        except ljm.LJMError:
            self.logger_box.module_logger.info("Failed config buffer stream")
            self.close_labjack_connection()
            raise

    # Set up trigger for buffer stream:
    def configure_stream_trigger(self):
        # https://labjack.com/pages/support?doc=/datasheets/t-series-datasheet/132-dio-extended-features-t-series-datasheet/
        # self.logger_box.module_logger.info("Configuring trigger")

        ljm.eWriteName(
            self.handle, "STREAM_TRIGGER_INDEX", 0
        )  # disabling triggered stream, also clears previous settings i think
        ljm.eWriteName(
            self.handle, "STREAM_CLOCK_SOURCE", 0
        )  # Enabling internally-clocked stream.
        ljm.eWriteName(self.handle, "STREAM_RESOLUTION_INDEX", 0)
        ljm.eWriteName(self.handle, "STREAM_SETTLING_US", 0)
        ljm.eWriteName(self.handle, "AIN_ALL_RANGE", 0)
        ljm.eWriteName(self.handle, "AIN_ALL_NEGATIVE_CH", ljm.constants.GND)
        # ----
        # Configure LJM for unpredictable stream timing. By default, LJM will time out with an error while waiting for the stream trigger to occur.
        ljm.writeLibraryConfigS(
            ljm.constants.STREAM_SCANS_RETURN,
            ljm.constants.STREAM_SCANS_RETURN_ALL_OR_NONE,
        )
        ljm.writeLibraryConfigS(ljm.constants.STREAM_RECEIVE_TIMEOUT_MS, 0)
        # ----
        # Define which address trigger is. Example:  2000 sets DIO0 / FIO0 as the stream trigger
        ljm.eWriteName(
            self.handle, "STREAM_TRIGGER_INDEX", ljm.nameToAddress(self.tr_sink_addr)[0]
        )
        # ----
        # CONFIGS FOR TRIGGERED STREAM USING Extended Feature INDEX 12 "CONDITIONAL RESET":    (DIO2_EF_CONFIG_B,  DIO2_EF_CONFIG_C not needed)
        # Clear any previous settings on triggerName's Extended Feature registers. Must be value 0 during configuration
        ljm.eWriteName(self.handle, "%s_EF_ENABLE" % self.tr_sink_addr, 0)
        # Choose which extended feature to set
        ljm.eWriteName(self.handle, "%s_EF_INDEX" % self.tr_sink_addr, 12)
        # Set reset options, see bitmask options
        ljm.eWriteName(
            self.handle, "%s_EF_CONFIG_A" % self.tr_sink_addr, 0
        )  # 0: Falling edges , 1: Rising edges (<-i think, depends on bitmask)
        # Turn on the DIO-EF  --> Enable trigger once configs are done
        ljm.eWriteName(self.handle, "%s_EF_ENABLE" % self.tr_sink_addr, 1)

        # Arming/loading trigger. Trigger activates when self.tr_source_addr goes from 1 to 0 --> falling edge trigger
        # ljm.eWriteName(self.handle, self.tr_source_addr, 1)  --> moved to command list!

    # Step 8) Sets start scan positions of galvos
    def init_start_positions(self):
        if abs(self.step_values_up[0]) < 5 and abs(self.sine_values[0]) < 5:
            ljm.eWriteNames(
                self.handle,
                2,
                [self.step_addr, self.sine_addr],
                [self.step_values_up[0], self.sine_values[0]],
            )
            # self.logger_box.module_logger.info("Setting start positions for Step (up) and Sine values:", self.step_values_up[0], ", ", self.sine_values[0])
        else:
            self.abort_scan = True

    def test_trigger(self):
        if self.useTrigger:
            self.logger_box.module_logger.info("")
            self.logger_box.module_logger.info("-------")
            self.logger_box.module_logger.info(f"Stream activated, but waiting. ")
            self.logger_box.module_logger.info(
                f"You can trigger stream now via a falling edge on {self.tr_source_addr}.\n"
            )
            self.logger_box.module_logger.info("Sleeping 3 seconds to test trigger:")
            for i in range(1, 3):
                self.logger_box.module_logger.info(i, "s ...")
                windowgui.root.after(int(1 * 1000), windowgui.sleep)

    # Step 9) Actual scan is done here
    def multi_start_scan(self):
        # Start configured (but trigger-set) stream --> scan several frames
        try:

            if self.abort_scan or self.offline:  # last line of defense
                self.logger_box.module_logger.info("Aborted scan (error or offline)")
                return

            self.logger_box.module_logger.info("Initializing start position")
            self.init_start_positions()  # TODO later, consider moving galvo a bit at start up for best results
            # windowgui.root.after(int(1 * 1000), windowgui.sleep)
            time.sleep(1)  # Note: give galvo a bit of time to reach start pos??

            self.logger_box.module_logger.info("Starting Scan")
            self.gui.root.update()
            # -----temp----
            # print("-----")
            # for i, _ in enumerate(self.aAddressesUp):
            #    print(self.aAddressesUp[i], "-->", self.aValuesUp[i])
            # print("-----")

            # ----end temp----

            start_time = time.time()
            # proc_step =
            for i in range(self.num_frames):  # scan repeats for given number of frames
                # self.gui.pb['value'] += proc_step
                # self.gui.root.update()  # testing    # TODO NOTE FIXME, CHECK IF THIS AFFECTS ANYTHING TIME-WISE!!
                # self.logger_box.module_logger.info(f"Frame", i, "done")

                if i % 2 == 0:  # if i is even
                    rc1 = ljm.eWriteNames(
                        self.handle,
                        len(self.aAddressesUp),
                        self.aAddressesUp,
                        self.aValuesUp,
                    )  # step left to right (or bottom to top)
                else:
                    rc2 = ljm.eWriteNames(
                        self.handle,
                        len(self.aAddressesDown),
                        self.aAddressesDown,
                        self.aValuesDown,
                    )  # step right to left (or top to bottom)

            end_time = time.time()

            err = ljm.eStreamStop(self.handle)
            self.logger_box.module_logger.info("Stream closed")
            self.logger_box.module_logger.info(
                f"Scan Done!"
                f"\n   ETA scan time = {int(self.scanTime)} seconds"
                f"\n   Theoretical scan time = {self.num_frames * self.step_dim * self.step_delay} seconds"
                f"\n   Actual scan time   = {round(end_time - start_time, 6)} seconds\n"
            )

            self.gui.root.update()
            # windowgui.root.after(int(1 * 1000), windowgui.sleep)
            self.socket_connection(doneScan=True)

            time.sleep(2)

            # reset trigger and galvo positions to offset:
            rc = ljm.eWriteName(
                self.handle, self.tr_source_addr, 0
            )  # send 0 just in case to stop any input
            self.set_offset_pos()

            # Tells server that we are done scanning
            self.logger_box.module_logger.info("Stopping")

        except ljm.LJMError:
            self.logger_box.module_logger.info("Failed scan")
            err = ljm.eStreamStop(self.handle)
            self.close_labjack_connection()

            raise

    # Sets galvos to set offset positions
    def set_offset_pos(self):
        ljm.eWriteNames(
            self.handle,
            2,
            [self.x_address, self.y_address],
            [self.x_offset, self.y_offset],
        )

    def print_log(self, txt, printlog=True):
        if printlog:
            self.logger_box.module_logger.info(txt)
        else:
            print("***", txt)

    # Terminates labjack connection
    def close_labjack_connection(self, printlog=True, closeserver=False):

        self.print_log("Closing labjack connection...", printlog)
        if self.handle is None:
            self.print_log(
                "T7 was not opened and therefore doesn't need closing", printlog
            )
        else:
            # reset galvo positions to offset:
            self.set_offset_pos()

            # stop stream in case it was active  # TODO: check if stopping a stream that is not active raises an error
            # err = ljm.eStreamStop(self.handle)

            # clear trigger source voltage:
            ljm.eWriteName(
                self.handle, self.tr_source_addr, 0
            )  # send 0 just in case to stop any input

            # close connection
            time.sleep(1)
            err = ljm.close(self.handle)

            if err is None:
                self.print_log("Closing successful.", printlog)
            else:
                self.print_log(f"Problem closing T7 device. Error = {err}", printlog)

        if closeserver:
            self.socket_connection(shutdown_server=True)


class SafetyTests:
    def __init__(self, gui, t7):
        self.logger_box = gui.logger_box
        self.tt7t = t7

    def check_voltages(self):
        # max is 5V but this gives a bit of margin, NOTE: val = 0.22*optical angle --> val = 1V is big enough for our scope
        max_voltage = 4
        t7 = self.tt7t

        # Checking that max allowed voltage is not changed. 5V is the absolute maximum allowed, but we give some margins
        if max_voltage > 4.5:
            self.logger_box.module_logger.info(
                "Error: to high max voltage, change back to 4V or consult script author"
            )
            t7.abort_scan = True

        for step in t7.step_values:
            # CHECKING INPUT VALUES TO SERVO, MAX ALLOWED IS 5V, WE HAVE 4V FOR MARGINS
            if abs(step) > max_voltage:
                self.logger_box.module_logger.info(
                    f"Error: Too large voltage ({step}V) found in step list!"
                )
                t7.abort_scan = True

        for val in t7.sine_values:
            # CHECKING INPUT VALUES TO SERVO, MAX ALLOWED IS 5V, WE HAVE 4V FOR MARGINS
            if abs(val) > max_voltage:
                self.logger_box.module_logger.info(
                    f"Error: Too large voltage ({val}V) found in sine list!"
                )
                self.tt7t.abort_scan = True
            # CHECKING INPUT VALUES TO SENT VIA DAC, ONLY POSITIVE VALUES ALLOWED
            if val <= 0:
                self.logger_box.module_logger.info(
                    f"Error: Negative voltage ({val}V) found in list for DAC!"
                )
                t7.abort_scan = True

    # MULTI-DONE
    def multi_check_cmd_list(
        self, addresses, values, check_txt=""
    ):  # check_cmd_list(self):
        # self.logger_box.module_logger.info(check_txt)

        # 1) WE CHECK THE COMMAND LIST LENGTHS
        t7 = self.tt7t

        if len(addresses) != len(values):
            self.logger_box.module_logger.info(
                "ERROR. NOT SAME COMMAND LIST LENGTHS. MISALIGNMENT DANGER."
            )
            t7.abort_scan = True

        # 2) WE CHECK THE STEPS  # TODO: maybe fix so we don't repeat this several times. although fix later
        for step in range(t7.step_dim):
            if t7.step_values[step] > 4:
                self.logger_box.module_logger.info(
                    "ERROR. STEP VALUE TOO LARGE:", t7.step_values[step]
                )
                t7.abort_scan = True
            if t7.step_values_up[step] > 4:
                self.logger_box.module_logger.info(
                    "ERROR. STEP UP VALUE TOO LARGE:", t7.step_values_up[step]
                )
                t7.abort_scan = True
            if t7.step_values_down[step] > 4:
                self.logger_box.module_logger.info(
                    "ERROR. STEP DOWN VALUE TOO LARGE:", t7.step_values_down[step]
                )
                t7.abort_scan = True

        if len(t7.step_values) != t7.step_dim:
            self.logger_box.module_logger.info(
                "ERROR. NOT ENOUGH STEP VALUES.", len(t7.step_values), "!=", t7.step_dim
            )
            t7.abort_scan = True
        if len(t7.step_values_up) != t7.step_dim:
            self.logger_box.module_logger.info(
                "ERROR. NOT ENOUGH STEP VALUES UP. ",
                len(t7.step_values_up),
                "!=",
                t7.step_dim,
            )
            t7.abort_scan = True
        if len(t7.step_values_down) != t7.step_dim:
            self.logger_box.module_logger.info(
                "ERROR. NOT ENOUGH STEP VALUES DOWN.",
                len(t7.step_values_down),
                "!=",
                t7.step_dim,
            )
            t7.abort_scan = True

        # 3) WE CHECK THE ADDRESSES IN
        for i in range(len(addresses)):
            if addresses[i] == t7.tr_source_addr:
                if values[i] != 0 and values[i] != 1:
                    self.logger_box.module_logger.info(
                        "ERROR. INVALID VALUE FOR EDGE SOURCE VALUE"
                    )
                    t7.abort_scan = True

            elif addresses[i] == t7.tr_sink_addr:
                self.logger_box.module_logger.info(
                    "ERROR. SINK SHOULD NOT BE A COMMAND TARGET ADDRESS"
                )
                t7.abort_scan = True

            elif addresses[i] == t7.wait_address:
                if values[i] < 100 and values[i] != 0:
                    # self.logger_box.module_logger.info("ERROR. ", values[i], " WAIT VALUE IS TOO SMALL.")
                    # t7.abort_scan = True
                    pass

            elif addresses[i] == t7.step_addr:
                if abs(values[i]) > 4:
                    self.logger_box.module_logger.info("ERROR. VALUE TOO BIG")
                    t7.abort_scan = True

            elif addresses[i] == t7.sine_addr:
                self.logger_box.module_logger.info("ERROR. SINE VALUE IN COMMAND LIST")
                t7.abort_scan = True
                if abs(values[i]) > 4:
                    self.logger_box.module_logger.info("ERROR. VALUE TOO BIG")

            elif (addresses[i] == t7.q_M101_addr) or (addresses[i] == t7.q_M102_addr):
                if values[i] != 0 and values[i] != 1:
                    self.logger_box.module_logger.info(
                        "ERROR. MARKER VALUE ERROR. MUST BE IN {0,1}"
                    )
                    t7.abort_scan = True

            elif addresses[i] == "STREAM_ENABLE" or addresses[i] == "STREAM_NUM_SCANS":
                pass
            else:
                self.logger_box.module_logger.info(
                    f"'{addresses[i]}' ... Address not recognized or checked for in 'check_cmd_list()'. Aborting scan."
                )

                t7.abort_scan = True

        if t7.abort_scan:
            self.logger_box.module_logger.info("Final Check Failed...\n")
        else:
            pass
            # self.logger_box.module_logger.info("Final Check Succeeded!\n")


# CORE OF CODE:
def main():
    global windowgui

    windowgui = WindowGUI()
    windowgui.root.mainloop()


main()
