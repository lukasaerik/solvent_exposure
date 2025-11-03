import sys, os, math
import pandas as pd
from biopandas.pdb import PandasPdb
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer, QEventLoop, QEvent
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPalette, QStandardItem, QFontMetrics
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from funcs import preprocess, f2_cutoff, exposure, create_3_vectors, average_score, score_v_localres, features

basedir = os.path.dirname(__file__)
standard_residues = ['LYS', 'LEU', 'THR', 'TYR', 'PRO', 'GLU', 'ASP', 'ILE', 'ALA', 'PHE', 'ARG',
                     'VAL', 'GLN', 'GLY', 'SER', 'TRP', 'CYS', 'HIS', 'ASN', 'MET', 'SEC', 'PYL']

class CheckableComboBox(QComboBox):

    # Subclass Delegate to increase item height
    class Delegate(QStyledItemDelegate):
        def sizeHint(self, option, index):
            size = super().sizeHint(option, index)
            size.setHeight(20)
            return size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Make the combo editable to set a custom text, but readonly
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        # Make the lineedit the same color as QPushButton
        palette = QApplication.palette()
        palette.setBrush(QPalette.Base, palette.button())
        self.lineEdit().setPalette(palette)

        # Use custom delegate
        self.setItemDelegate(CheckableComboBox.Delegate())

        # Update the text when an item is toggled
        self.model().dataChanged.connect(self.updateText)

        # Hide and show popup when clicking the line edit
        self.lineEdit().installEventFilter(self)
        self.closeOnLineEditClick = False

        # Prevent popup from closing when clicking on an item
        self.view().viewport().installEventFilter(self)

    def resizeEvent(self, event):
        # Recompute text to elide as needed
        self.updateText()
        super().resizeEvent(event)

    def eventFilter(self, object, event):

        if object == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
        return False

    def showPopup(self):
        super().showPopup()
        # When the popup is displayed, a click on the lineedit should close it
        self.closeOnLineEditClick = True

    def hidePopup(self):
        super().hidePopup()
        # Used to prevent immediate reopening when clicking on the lineEdit
        self.startTimer(100)
        # Refresh the display text when closing
        self.updateText()

    def timerEvent(self, event):
        # After timeout, kill timer, and reenable click on line edit
        self.killTimer(event.timerId())
        self.closeOnLineEditClick = False

    def updateText(self):
        texts = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, texts, datalist=None):
        for i, text in enumerate(texts):
            try:
                data = datalist[i]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)

    def currentData(self):
        # Return the list of selected items data
        res = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                res.append(self.model().item(i).data())
        return res

class MplCanvas(FigureCanvas):
    """Simple Matplotlib FigureCanvas to hold one Axes."""
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        # let Qt resize the canvas nicely
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

class MatplotlibWidget(QWidget):
    """A QWidget containing a Matplotlib canvas and the standard toolbar."""
    def __init__(self, parent=None, initial_image=None):
        super().__init__(parent)
        # self.settings = settings
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        if initial_image is None:
            initial_image = [[0,5,10,15,20], [0.5,0.4,0.3,0.2,0.1]]

        self.canvas.ax.plot(*initial_image)

        self._update_reciprocal_y_ticks()      # apply once immediately
        self._connect_tick_callbacks_once()

        # ymin, ymax = self.canvas.ax.get_ylim()
        
        # ticks = reciprocal_ticks(ymin, ymax)

        # self.canvas.ax.set_yticks(1/ticks)
        # self.canvas.ax.set_yticklabels(f'1/{tick:.2g}' for tick in ticks)

        self.canvas.figure.tight_layout()

        # self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.canvas.draw_idle()

    def update_image(self, array):
        self.canvas.ax.plot(*array)

        # ymin, ymax = self.canvas.ax.get_ylim()
        
        # ticks = reciprocal_ticks(ymin, ymax)

        # self.canvas.ax.set_yticks(1/ticks)
        # self.canvas.ax.set_yticklabels(f'1/{tick:.2g}' for tick in ticks)

        self.canvas.draw_idle()

    def reciprocal_ticks(self, mn, mx, n = 4, intervals = [1, 2, 5, 10, 20, 50, 100]):
        # your function, unchanged except ensure mn < mx and positive
        ticks = []
        if mn == mx:
            return np.array([])
        # ensure mn < mx
        low = min(mn, mx)
        high = max(mn, mx)
        # require positive region for reciprocals
        if high <= 0:
            return np.array([])
        # clamp low to a small positive if it crosses zero
        if low <= 0:
            low = 1e-12
        for i in intervals:
            # check whether this interval can provide enough ticks across [low, high]
            if i/low - i/high < n:
                # not enough density for this interval, continue searching
                continue
            else:
                # start at ceil(i/low) / i and step by -1/i until <= 1/high
                tick = math.ceil(i/low) / i
                while tick > 1/high:
                    ticks.append(tick)
                    tick -= 1.0 / i
                break
        return np.array(sorted(ticks))

    # Place this inside your widget/class where ax and canvas are available.
    def _update_reciprocal_y_ticks(self, ax=None):
        """Compute ticks from current y-limits and apply them as 1/x labels."""
        if ax is None:
            ax = self.canvas.ax
        ymin, ymax = ax.get_ylim()
        # we assume y values are positive (1/Å). If not, fall back to formatter only.
        if ymax <= 0:
            # fallback: just format existing ticks as 1/(1/y) but can't do negative/zero
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y:g}"))
            self.canvas.draw_idle()
            return

        # compute ticks in the current data space
        ticks = self.reciprocal_ticks(ymin, ymax)
        if ticks.size == 0:
            # no nice reciprocal ticks found — keep existing ticks but format them
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"1/{1/y:.3g}" if y != 0 else "0"))
        else:
            ax.set_yticks(1/ticks)
            # set labels as 1/<denominator> where denominator is 1/y
            labels = []
            for val in ticks:
                if val == 0:
                    labels.append("0")
                else:
                    denom = val
                    # pretty formatting: remove trailing zeros, but keep .5 if present
                    if abs(denom - round(denom)) < 1e-8:
                        labels.append(f"1/{int(round(denom))}")
                    else:
                        labels.append(f"1/{denom:g}")
            ax.set_yticklabels(labels)

        self.canvas.draw_idle()

    # Call this once after you create the axes to hook callbacks (e.g., in plot_score_vs_resolution)
    def _connect_tick_callbacks_once(self):
        ax = self.canvas.ax
        # avoid connecting multiple times
        if getattr(self, "_recip_callbacks_connected", False):
            return
        # connect to ylim_changed (passes the Axes object as argument)
        ax.callbacks.connect('ylim_changed', lambda ax: self._update_reciprocal_y_ticks(ax))
        # also connect to draw_event on the figure as a fallback (passes event)
        self.canvas.mpl_connect('draw_event', lambda event: self._update_reciprocal_y_ticks())
        self._recip_callbacks_connected = True

    # def get_current_image(self):
    #     """Return current image array (a view/copy depending on backend)."""
    #     return self.im.get_array()

    # def on_click(self, event):
    #     # event.xdata, event.ydata may be None if click was outside axes
    #     if event.inaxes is self.canvas.ax:
    #         print(f"Mouse click at x={event.xdata:.3f}, y={event.ydata:.3f}, button={event.button}")

    # def on_scroll(self, event):
    #     # example: zoom in/out centered at mouse pos
    #     base_scale = 1.1
    #     if event.inaxes is None:
    #         return
    #     ax = event.inaxes
    #     cur_xlim = ax.get_xlim()
    #     cur_ylim = ax.get_ylim()
    #     xdata = event.xdata
    #     ydata = event.ydata
    #     if event.button == 'up':
    #         # zoom in
    #         scale_factor = 1 / base_scale
    #     else:
    #         # zoom out
    #         scale_factor = base_scale
    #     new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    #     new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
    #     ax.set_xlim([xdata - new_width * (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]),
    #                  xdata + new_width * (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])])
    #     ax.set_ylim([ydata - new_height * (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]),
    #                  ydata + new_height * (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])])
        
    #     ymin, ymax = self.canvas.ax.get_ylim()
        
    #     ticks = reciprocal_ticks(ymin, ymax)

    #     ax.set_yticks(1/ticks)
    #     ax.set_yticklabels(f'1/{tick}' for tick in ticks)
        
    #     self.canvas.draw_idle()

    def plot_score_vs_resolution(self, x, y, names=None, xlabel=None, ylabel=None):
        # clear old plot
        self.canvas.ax.clear()
        sc = self.canvas.ax.scatter(x, y, s=10, picker=True)

        if xlabel:
            self.canvas.ax.set_xlabel(xlabel)
        if ylabel:
            self.canvas.ax.set_ylabel(ylabel)

        # create annotation on the canvas (main thread)
        annot = self.canvas.ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                                    bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}".format(" ".join([names[n] for n in ind["ind"]])) if names else ""
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.8)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == self.canvas.ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    self.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        self.canvas.draw_idle()

        
        
        
        # connect on the FigureCanvas instance
        self.canvas.mpl_connect("motion_notify_event", hover)

        # ymin, ymax = self.canvas.ax.get_ylim()
        
        # ticks = reciprocal_ticks(ymin, ymax)

        # self.canvas.ax.set_yticks(1/ticks)
        # self.canvas.ax.set_yticklabels(f'1/{tick}' for tick in ticks)

        self._update_reciprocal_y_ticks()      # apply once immediately
        self._connect_tick_callbacks_once()
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()

class ScriptWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(object)   # can emit results
    error = pyqtSignal(str)

    ask = pyqtSignal(str)
    answer = pyqtSignal(bool)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings  # dict of settings

    def run_simple(self):
        self.started.emit()
        try:
            settings = self.settings
            
            pdb_path = settings.get("pdb_path")
            pre_path = settings.get("folder_pre_path")
            out_path = settings.get("folder_out_path")

            pre_out = preprocess(pdb_path=pdb_path, pre_path=pre_path, yes_no=self.yes_no)
            result = exposure(pdb_path=pre_out, out_path=out_path)

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def run_adduct_pre(self):
        self.started.emit()
        try:
            settings = self.settings
            pdb_path = settings.get("pdb_path")
            pre_path = settings.get("folder_pre_path")
            feature = settings.get("feature")

            pre_out = preprocess(pdb_path=pdb_path, pre_path=pre_path, yes_no=self.yes_no)
            features_out = features(pdb_path=pre_out, feature=feature)

            result = pre_out, features_out
            self.finished.emit(result)
        # except UserWarning:
        #     settings = self.settings
        #     pdb_path = settings.get("pdb_path")
        #     pre_path = settings.get("folder_pre_path")
        #     feature = settings.get("feature")

        #     pre_out = preprocess(pdb_path=pdb_path, pre_path=pre_path, yes_no=self.yes_no)
        #     features_out = features(pdb_path=pre_out, feature=feature)

        #     result = pre_out, features_out
        #     self.finished.emit(result)            
        except Exception as e:
            self.error.emit(str(e))

    def run_adduct_out(self):
        self.started.emit()
        try:
            settings = self.settings
            pdb_path = settings.get("pre_out_path")
            out_path = settings.get("folder_out_path")
            feature = settings.get("feature")
            combo = settings.get("combo")

            assignment = create_3_vectors(pdb_path=pdb_path, chain1=combo, feature=feature)
            result = exposure(pdb_path=pdb_path, out_path=out_path, assignment=assignment)

            self.finished.emit(result)
        # except UserWarning:
        #     settings = self.settings
        #     pdb_path = settings.get("pdb_path")
        #     pre_path = settings.get("folder_pre_path")
        #     feature = settings.get("feature")

        #     pre_out = preprocess(pdb_path=pdb_path, pre_path=pre_path, yes_no=self.yes_no)
        #     features_out = features(pdb_path=pre_out, feature=feature)

        #     result = pre_out, features_out
        #     self.finished.emit(result)            
        except Exception as e:
            self.error.emit(str(e))

    def run_plot(self):
        self.started.emit()
        try:
            settings = self.settings
            pdb_path = settings.get("pdb_path")
            defattr_path = settings.get("defattr_path")
            only_chain = settings.get("only_chain")
            # compute data (no plotting)
            result = score_v_localres(pdb_path=pdb_path, defattr_path=defattr_path, only_chain=only_chain, called_by_GUI=True, inverse=True)
            # emit the computed dict
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        
    def yes_no(self, text):
        """
        Called from worker thread. Will emit `ask` (handled in main thread)
        and wait for `answer` to be emitted by main thread.
        """
        loop = QEventLoop()
        response_holder = {'val': None}

        def _on_answer(val):
            response_holder['val'] = val
            loop.quit()

        # connect temporary slot
        self.answer.connect(_on_answer)
        # emit request to main thread
        self.ask.emit(text)
        # block worker thread until main thread answers
        loop.exec_()
        # cleanup connection
        try:
            self.answer.disconnect(_on_answer)
        except Exception:
            pass

        return bool(response_holder['val'])




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solvent Exposure Calculation")

        # Set up Tabs
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setMovable(False)

        ###
        # Simple Tab
        ###
        simple = QWidget()
        simple_form = QFormLayout()

        self.current_simple_settings = {
            'pdb_path': os.path.join(basedir, 'pdbs', 'in', '1u7g.pdb'),
            'folder_pre_path': os.path.join(basedir, 'pdbs', 'preprocessed'),
            'folder_out_path': os.path.join(basedir, 'pdbs', 'out'),
        }

        self.interactive = True
        self.only_chain = False

        # PDB file selection
        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setText(self.current_simple_settings.get("pdb_path", ""))
        file_browse = QPushButton("Browse...")
        file_browse.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_edit)
        file_row.addWidget(file_browse)
        simple_form.addRow("PDB File:", file_row)

        # Folder (preproccessed) selection
        folder_pre_row = QHBoxLayout()
        self.folder_pre_edit = QLineEdit()
        self.folder_pre_edit.setText(self.current_simple_settings.get("folder_pre_path", ""))
        folder_pre_browse = QPushButton("Browse...")
        folder_pre_browse.clicked.connect(self._browse_pre_folder)
        folder_pre_row.addWidget(self.folder_pre_edit)
        folder_pre_row.addWidget(folder_pre_browse)
        simple_form.addRow("Preprocessed Folder:", folder_pre_row)

        # Folder (out) selection
        folder_out_row = QHBoxLayout()
        self.folder_out_edit = QLineEdit()
        self.folder_out_edit.setText(self.current_simple_settings.get("folder_out_path", ""))
        folder_out_browse = QPushButton("Browse...")
        folder_out_browse.clicked.connect(self._browse_out_folder)
        folder_out_row.addWidget(self.folder_out_edit)
        folder_out_row.addWidget(folder_out_browse)
        simple_form.addRow("Output Folder:", folder_out_row)
        
        # Output text box
        self.simple_output = QTextEdit()
        self.simple_output.setReadOnly(True)
        self.simple_output.setPlaceholderText("Results will appear here...")
        simple_form.addRow('', self.simple_output)

        # Bottom: Run Button
        self.run_simple = QPushButton("Calculate")
        # self.run_simple.setIcon(QIcon(os.path.join(basedir, "icons", "blast.png")))
        self.run_simple.clicked.connect(self.on_run_simple_clicked)
        simple_form.addRow('', self.run_simple)

        # Add to tab
        simple.setLayout(simple_form)
        tabs.addTab(simple, 'Simple')


        ###
        # Adduct Tab
        ###
        adduct = QWidget()
        adduct_form = QFormLayout()

        self.current_adduct_settings = {
            'pdb_path': os.path.join(basedir, 'pdbs', 'in', '1u7g.pdb'),
            'folder_pre_path': os.path.join(basedir, 'pdbs', 'preprocessed'),
            'feature': 'chain_id',
            'pre_out_path': '',
            'folder_out_path': os.path.join(basedir, 'pdbs', 'out'),
            'combo': '',
        }

        # PDB File selection
        adduct_file_row = QHBoxLayout()
        self.adduct_file_edit = QLineEdit()
        self.adduct_file_edit.setText(self.current_adduct_settings.get("pdb_path", ""))
        adduct_file_browse = QPushButton("Browse...")
        adduct_file_browse.clicked.connect(self._browse_adduct_file)
        adduct_file_row.addWidget(self.adduct_file_edit)
        adduct_file_row.addWidget(adduct_file_browse)
        adduct_form.addRow("PDB File:", adduct_file_row)

        # Folder (preprocessed) selection
        adduct_folder_pre_row = QHBoxLayout()
        self.adduct_folder_pre_edit = QLineEdit()
        self.adduct_folder_pre_edit.setText(self.current_adduct_settings.get("folder_pre_path", ""))
        adduct_folder_pre_browse = QPushButton("Browse...")
        adduct_folder_pre_browse.clicked.connect(self._browse_adduct_pre_folder)
        adduct_folder_pre_row.addWidget(self.adduct_folder_pre_edit)
        adduct_folder_pre_row.addWidget(adduct_folder_pre_browse)
        adduct_form.addRow("Preprocessed Folder:", adduct_folder_pre_row)

        # Feature selection
        self.adduct_feature = QComboBox()
        self.adduct_feature.addItems(['chain_id', 'residue_name'])
        op = self.current_adduct_settings.get('feature', 'chain_id')
        idx = self.adduct_feature.findText(op)
        if idx >= 0:
            self.adduct_feature.setCurrentIndex(idx)
        adduct_form.addRow("Feature", self.adduct_feature)

        # Preprocess/feature Run Button
        self.run_adduct_pre = QPushButton("Calculate")
        self.run_adduct_pre.clicked.connect(self.on_run_adduct_pre_clicked)
        adduct_form.addRow('', self.run_adduct_pre)

        # Combo selector
        self.combo = CheckableComboBox()
        adduct_form.addRow('Combo', self.combo)

        # Folder (out) selection
        adduct_folder_out_row = QHBoxLayout()
        self.adduct_folder_out_edit = QLineEdit()
        self.adduct_folder_out_edit.setText(self.current_adduct_settings.get("folder_out_path", ""))
        adduct_folder_out_browse = QPushButton("Browse...")
        adduct_folder_out_browse.clicked.connect(self._browse_adduct_out_folder)
        adduct_folder_out_row.addWidget(self.adduct_folder_out_edit)
        adduct_folder_out_row.addWidget(adduct_folder_out_browse)
        adduct_form.addRow("Output Folder:", adduct_folder_out_row)

        # Bottom: Run Button
        self.run_adduct_out = QPushButton("Calculate")
        self.run_adduct_out.clicked.connect(self.on_run_adduct_out_clicked)
        adduct_form.addRow('', self.run_adduct_out)

        # Output text box
        self.adduct_output = QTextEdit()
        self.adduct_output.setReadOnly(True)
        self.adduct_output.setPlaceholderText("Results will appear here...")
        adduct_form.addRow('', self.adduct_output)

        adduct.setLayout(adduct_form)
        tabs.addTab(adduct, 'With Adduct')


        ###
        # Manual Tab
        ###
        tabs.addTab(QLabel('not yet implemented'), 'Manual')


        ###
        # Plotting Tab
        ###
        plot = QWidget()
        plot_form = QFormLayout()

        self.current_plot_settings = {
            'pdb_path': os.path.join(basedir, 'pdbs', 'out', '3jcz_2c50_26p5.pdb'),
            'defattr_path': os.path.join(basedir, 'pdbs', 'out', 'defattrs', 'gdh_J123.defattr'),
            'interactive': True,
            'only_chain': False,
        }

        plot_pdb_row = QHBoxLayout()
        self.plot_pdb_edit = QLineEdit()
        self.plot_pdb_edit.setText(self.current_plot_settings.get("pdb_path", ""))
        plot_pdb_browse = QPushButton("Browse...")
        plot_pdb_browse.clicked.connect(self._browse_plot_file)
        plot_pdb_row.addWidget(self.plot_pdb_edit)
        plot_pdb_row.addWidget(plot_pdb_browse)
        plot_form.addRow("PDB File:", plot_pdb_row)

        plot_defattr_row = QHBoxLayout()
        self.plot_defattr_edit = QLineEdit()
        self.plot_defattr_edit.setText(self.current_plot_settings.get("defattr_path", ""))
        plot_defattr_browse = QPushButton("Browse...")
        plot_defattr_browse.clicked.connect(self._browse_defattr_file)
        plot_defattr_row.addWidget(self.plot_defattr_edit)
        plot_defattr_row.addWidget(plot_defattr_browse)
        plot_form.addRow("defattr File:", plot_defattr_row)

        # Only chain(s)?
        only_chains_h = QHBoxLayout()
        self.only_chain_checkbox = QCheckBox("Only Chain(s)?")
        self.only_chain_checkbox.setChecked(self.only_chain)
        self.only_chain_checkbox.stateChanged.connect(self._on_only_chains_toggled)
        only_chains_h.addWidget(self.only_chain_checkbox)
        self.only_chain_combo = CheckableComboBox()
        self.only_chain_combo.addItems(features(pdb_path=self.plot_pdb_edit.text(), feature='chain_id'))
        only_chains_h.addWidget(self.only_chain_combo)
        plot_form.addRow('', only_chains_h)

        # Interactive mode
        bottom_h = QHBoxLayout()
        self.interactive_checkbox = QCheckBox("Interactive?")
        self.interactive_checkbox.setChecked(self.interactive)
        self.interactive_checkbox.stateChanged.connect(self._on_interactive_toggled)
        bottom_h.addWidget(self.interactive_checkbox, alignment=Qt.AlignLeft)

        # Plot button
        self.run_plot = QPushButton("Plot")
        # self.run_plot.setIcon(QIcon(os.path.join(basedir, "icons", "blast.png")))
        self.run_plot.clicked.connect(self.on_run_plot_clicked)
        bottom_h.addWidget(self.run_plot)
        plot_form.addRow('', bottom_h)

        self.sc = MatplotlibWidget(self)
        plot_form.addWidget(self.sc)

        # Output text box
        self.plot_output = QTextEdit()
        self.plot_output.setReadOnly(True)
        self.plot_output.setPlaceholderText("Results will appear here...")
        plot_form.addRow('', self.plot_output)

        plot.setLayout(plot_form)
        tabs.addTab(plot, 'Score vs Resolution')

        self.setCentralWidget(tabs)

    def on_run_simple_clicked(self):
        # gather values
        # settings = tuple(spin.value() for spin in self.spin_boxes)
        # self.output.append(f"Starting script with values: {vals} and settings: {self.current_settings}")

        # disable run button while running
        self.run_simple.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_simple_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_simple)
        # self.worker.started.connect(lambda: self.output.append("Worker started..."))
        self.worker.finished.connect(self.on_worker_simple_finished)
        self.worker.error.connect(self.on_worker_simple_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.ask.connect(self._on_worker_ask_question)

        # start the thread
        self.thread.start()

    def on_worker_simple_finished(self, result):
        # re-enable run button
        self.run_simple.setEnabled(True)
        for i in result:
            self.simple_output.append(f"File {i[0]} saved. \n Min: {i[1]} \n Max: {i[2]}")

    def on_worker_simple_error(self, err_str):
        self.run_simple.setEnabled(True)
        self.simple_output.append(f"Worker error: {err_str}")
        QMessageBox.critical(self, "Script error", f"An error occurred:\n{err_str}")

    def on_run_adduct_pre_clicked(self):
        # gather values
        updated_settings = self.get_adduct_settings()
        for k in updated_settings:
            self.current_adduct_settings[k] = updated_settings.get(k)

        # disable run button while running
        self.run_adduct_pre.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_adduct_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_adduct_pre)
        # self.worker.started.connect(lambda: self.output.append("Worker started..."))
        self.worker.finished.connect(self.on_worker_adduct_pre_finished)
        self.worker.error.connect(self.on_worker_adduct_pre_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.ask.connect(self._on_worker_ask_question)

        # start the thread
        self.thread.start()

    def on_worker_adduct_pre_finished(self, result):
        # re-enable run button
        self.run_adduct_pre.setEnabled(True)
        pre_out, options = result
        self.current_adduct_settings["pre_out_path"] = pre_out
        self.adduct_output.append(f"File {pre_out} saved. \n There were {len(options)} unique entries under {self.current_adduct_settings.get("feature")}.")
        self.combo.clear()
        self.combo.addItems(options)

    def on_worker_adduct_pre_error(self, err_str):
        self.run_adduct_pre.setEnabled(True)
        # self.adduct_pre_output.append(f"Worker error: {err_str}")
        QMessageBox.critical(self, "Script error", f"An error occurred:\n{err_str}")

    def on_run_adduct_out_clicked(self):
        # gather values
        updated_settings = self.get_adduct_settings()
        for k in updated_settings:
            self.current_adduct_settings[k] = updated_settings.get(k)

        # disable run button while running
        self.run_adduct_out.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_adduct_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_adduct_out)
        # self.worker.started.connect(lambda: self.output.append("Worker started..."))
        self.worker.finished.connect(self.on_worker_adduct_out_finished)
        self.worker.error.connect(self.on_worker_adduct_out_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.ask.connect(self._on_worker_ask_question)

        # start the thread
        self.thread.start()

    def on_worker_adduct_out_finished(self, result):
        # re-enable run button
        self.run_adduct_out.setEnabled(True)
        for i in result:
            self.adduct_output.append(f"File {i[0]} saved. \n Min: {i[1]} \n Max: {i[2]}")

    def on_worker_adduct_out_error(self, err_str):
        self.run_adduct_out.setEnabled(True)
        # self.adduct_out_output.append(f"Worker error: {err_str}")
        QMessageBox.critical(self, "Script error", f"An error occurred:\n{err_str}")

    def _on_interactive_toggled(self, state):
        self.interactive = bool(state)
        # reflect in current_plot_settings immediately
        self.current_plot_settings["interactive"] = self.interactive

    def _on_only_chains_toggled(self, state):
        self.only_chain = bool(state)
        # reflect in current_plot_settings immediately
        self.current_plot_settings["only_chain"] = self.only_chain

    def on_run_plot_clicked(self):
        # gather values
        updated_settings = self.get_plot_settings()
        self.plot_output.append(f'{updated_settings}')
        for k in updated_settings:
            self.current_plot_settings[k] = updated_settings.get(k)

        # disable run button while running
        self.run_plot.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_plot_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_plot)
        # self.worker.started.connect(lambda: self.output.append("Worker started..."))
        self.worker.finished.connect(self.on_worker_plot_finished)
        self.worker.error.connect(self.on_worker_plot_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.ask.connect(self._on_worker_ask_question)

        # start the thread
        self.thread.start()

    def on_worker_plot_finished(self, result):
        # re-enable run button
        self.run_plot.setEnabled(True)

        # result is the dict emitted by worker
        if isinstance(result, dict):
            x = result['x']
            y = result['y']
            names = result.get('names', None)
            xlabel = result.get('xlabel', None)
            ylabel = result.get('ylabel', None)

            # sc is the MatplotlibWidget instance you created earlier
            # ensure you stored it as an attribute so you can access it here:
            try:
                self.sc.plot_score_vs_resolution(x, y, names=names, xlabel=xlabel, ylabel=ylabel)
            except Exception as e:
                QMessageBox.critical(self, "Plot error", f"Could not draw plot:\n{e}")
        else:
            # fallback text output if something unexpected
            self.plot_output.append('Plot worker finished (unexpected result format).')

    def on_worker_plot_error(self, err_str):
        self.run_plot.setEnabled(True)
        self.plot_output.append(f"Worker error: {err_str}")
        QMessageBox.critical(self, "Script error", f"An error occurred:\n{err_str}")


    def _browse_file(self):
        # open file dialog to choose a file
        fname, _ = QFileDialog.getOpenFileName(self, "Select file", self.file_edit.text() or "", "All Files (*)")
        if fname:
            self.file_edit.setText(fname)

    def _browse_adduct_file(self):
        # open file dialog to choose a file
        fname, _ = QFileDialog.getOpenFileName(self, "Select file", self.adduct_file_edit.text() or "", "All Files (*)")
        if fname:
            self.adduct_file_edit.setText(fname)

    def _browse_plot_file(self):
        # open file dialog to choose a file
        fname, _ = QFileDialog.getOpenFileName(self, "Select file", self.plot_pdb_edit.text() or "", "All Files (*)")
        if fname:
            self.plot_pdb_edit.setText(fname)

    def _browse_defattr_file(self):
        # open file dialog to choose a file
        fname, _ = QFileDialog.getOpenFileName(self, "Select file", self.plot_defattr_edit.text() or "", "All Files (*)")
        if fname:
            self.plot_defattr_edit.setText(fname)

    def _browse_pre_folder(self):
        # open folder dialog
        folder = QFileDialog.getExistingDirectory(self, "Select folder", self.folder_pre_edit.text() or "")
        if folder:
            self.folder_pre_edit.setText(folder)
            
    def _browse_out_folder(self):
        # open folder dialog
        folder = QFileDialog.getExistingDirectory(self, "Select folder", self.folder_out_edit.text() or "")
        if folder:
            self.folder_out_edit.setText(folder)    
    
    def _browse_adduct_pre_folder(self):
        # open folder dialog
        folder = QFileDialog.getExistingDirectory(self, "Select folder", self.adduct_folder_pre_edit.text() or "")
        if folder:
            self.adduct_folder_pre_edit.setText(folder)
            
    def _browse_adduct_out_folder(self):
        # open folder dialog
        folder = QFileDialog.getExistingDirectory(self, "Select folder", self.adduct_folder_out_edit.text() or "")
        if folder:
            self.adduct_folder_out_edit.setText(folder)

    def _on_worker_ask_question(self, text):
        # This runs on the main thread (slot invoked in main thread)
        dlg = QMessageBox(self)
        dlg.setWindowTitle("User Input Required")
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dlg.setIcon(QMessageBox.Question)
        button = dlg.exec()
        yn = (button == QMessageBox.Yes)
        # send the answer back to the worker
        # worker.answer is a signal defined in worker; safe to emit from main thread
        self.worker.answer.emit(yn)

    def get_adduct_settings(self):
        # Return a dict of settings (strings where appropriate)
        return {
            "feature": self.adduct_feature.currentText(),
            "combo": self.combo.currentData(),
        }
    
    def get_plot_settings(self):
        # Return a dict of settings (strings where appropriate)
        if self.only_chain:
            return {
            "pdb_path": self.plot_pdb_edit.text(),
            "defattr_path": self.plot_defattr_edit.text(),
            'only_chain': self.only_chain_combo.currentData(),
        }
        else:
            return {
            "pdb_path": self.plot_pdb_edit.text(),
            "defattr_path": self.plot_defattr_edit.text(),
        }

    


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()