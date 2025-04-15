from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class ZClipWindow(ctk.CTkToplevel):
    def __init__(
            self,
            parent,
            config: TrainConfig,
            ui_state: UIState,
            *args, **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.config = config
        self.ui_state = ui_state
        self.zclip_ui_state = ui_state.get_var("zclip_config") # "zclip_config" を参照
        self.protocol("WM_DELETE_WINDOW", self.__ok) # ウィンドウを閉じるときは OK と同じ

        self.title("ZClip Settings")
        self.geometry("600x400") # サイズは適宜調整
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # --- UI 要素の配置 ---
        frame = self.__content_frame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        # ---

        components.button(self, 1, 0, "ok", self.__ok)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))


    def __content_frame(self, master):
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, minsize=50) # スペーサー
        master.grid_columnconfigure(3, weight=0)
        master.grid_columnconfigure(4, weight=1)

        row_index = 0

        # Alpha
        components.label(master, row_index, 0, "Alpha",
                         tooltip="Smoothing factor for EMA mean and variance (0 < alpha < 1).")
        components.entry(master, row_index, 1, self.zclip_ui_state, "alpha")
        row_index += 1

        # Z Threshold
        components.label(master, row_index, 0, "Z Threshold",
                         tooltip="Threshold for z-score or percentile calculation.")
        components.entry(master, row_index, 1, self.zclip_ui_state, "z_thresh")
        row_index += 1

        # Warmup Steps
        components.label(master, row_index, 0, "Warmup Steps",
                         tooltip="Number of steps to collect gradient norms before EMA initialization.")
        components.entry(master, row_index, 1, self.zclip_ui_state, "warmup_steps")
        row_index += 1

        # Mode
        components.label(master, row_index, 0, "Mode",
                         tooltip="Clipping mode: 'zscore' or 'percentile'.")
        components.options(master, row_index, 1, ["zscore", "percentile"], self.zclip_ui_state, "mode", command=self.__on_mode_change)
        row_index += 1

        # Clip Option (Mode が 'zscore' の場合のみ表示)
        self.clip_option_label = components.label(master, row_index, 0, "Clip Option (ZScore)",
                                                  tooltip="Action when z-score > threshold: 'adaptive_scaling' or 'mean'.")
        self.clip_option_widget = components.options(master, row_index, 1, ["adaptive_scaling", "mean"], self.zclip_ui_state, "clip_option")
        row_index += 1

        # Clip Factor (Mode が 'zscore' and Clip Option が 'adaptive_scaling' の場合のみ表示)
        self.clip_factor_label = components.label(master, row_index, 0, "Clip Factor (Adaptive)",
                                                 tooltip="Multiplier for adaptive scaling threshold (0.3-1.0 recommended).")
        self.clip_factor_widget = components.entry(master, row_index, 1, self.zclip_ui_state, "clip_factor")
        row_index += 1

        # Skip Update on Spike
        components.label(master, row_index, 0, "Skip EMA on Spike",
                         tooltip="If True, skip updating EMA statistics when a spike (clipping occurs) is detected.")
        components.switch(master, row_index, 1, self.zclip_ui_state, "skip_update_on_spike")
        row_index += 1

        # --- デフォルト値ロードボタン ---
        # Defaults Button (必要であれば)
        # components.label(master, row_index, 0, "Load Defaults",
        #                  tooltip="Load default ZClip settings")
        # components.button(master, row_index, 1, "Load Defaults", self.__load_defaults,
        #                   tooltip="Load default ZClip settings")
        # row_index += 1

        # --- 初期表示状態の更新 ---
        self.__update_visibility() # 初期表示のために呼び出す

    def __update_visibility(self):
        """Mode と Clip Option の選択に応じて UI 要素の表示/非表示を切り替える"""
        selected_mode = self.zclip_ui_state.get_var("mode").get()
        selected_clip_option = self.zclip_ui_state.get_var("clip_option").get()

        is_zscore_mode = selected_mode == "zscore"
        is_adaptive_scaling = selected_clip_option == "adaptive_scaling"

        # Clip Option の表示制御
        if is_zscore_mode:
            self.clip_option_label.grid(row=4, column=0, sticky="w")
            self.clip_option_widget.grid(row=4, column=1, sticky="ew")
        else:
            self.clip_option_label.grid_remove()
            self.clip_option_widget.grid_remove()

        # Clip Factor の表示制御
        if is_zscore_mode and is_adaptive_scaling:
            self.clip_factor_label.grid(row=5, column=0, sticky="w")
            self.clip_factor_widget.grid(row=5, column=1, sticky="ew")
        else:
            self.clip_factor_label.grid_remove()
            self.clip_factor_widget.grid_remove()

    def __on_mode_change(self, *args):
        """Mode が変更されたときに呼び出され、表示を更新"""
        self.__update_visibility()

    # def __load_defaults(self):
    #     """デフォルト値をロードする (必要であれば実装)"""
    #     default_config = TrainZClipConfig.default_values()
    #     self.zclip_ui_state.update(default_config.to_dict())
    #     self.__update_visibility() # デフォルトロード後も表示を更新

    def __ok(self):
        # 値の検証などが必要な場合はここで行う
        # 例: alpha が 0~1 の間か、warmup_steps が整数か等
        self.destroy()
