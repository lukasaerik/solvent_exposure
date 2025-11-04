import sys, os, math, time, mplcursors
import pandas as pd
from biopandas.pdb import PandasPdb
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QEventLoop, QEvent
from PyQt6.QtGui import QAction, QIcon, QPalette, QStandardItem, QFontMetrics, QKeySequence
from PyQt6.QtWidgets import (
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

from funcs import preprocess, create_3_vectors, exposure, score_v_localres, features

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
        palette.setBrush(QPalette.ColorRole.Base, palette.button())
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
            if event.type() == QEvent.Type.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return False

        if object == self.view().viewport():
            if event.type() == QEvent.Type.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.CheckState.Checked:
                    item.setCheckState(Qt.CheckState.Unchecked)
                else:
                    item.setCheckState(Qt.CheckState.Checked)
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
            if self.model().item(i).checkState() == Qt.CheckState.Checked:
                texts.append(self.model().item(i).text())
        text = ", ".join(texts)

        # Compute elided text (with "...")
        metrics = QFontMetrics(self.lineEdit().font())
        elidedText = metrics.elidedText(text, Qt.TextElideMode.ElideRight, self.lineEdit().width())
        self.lineEdit().setText(elidedText)

    def addItem(self, text, data=None):
        item = QStandardItem()
        item.setText(text)
        if data is None:
            item.setData(text)
        else:
            item.setData(data)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
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
            if self.model().item(i).checkState() == Qt.CheckState.Checked:
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
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None, initial_image=None):
        super().__init__(parent)
        self.setWindowTitle("Solvent Exposure Calculator")
        self.canvas = MplCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        if initial_image is None:
            initial_image = [[0, 5, 10, 15, 20], [0.5, 0.4, 0.3, 0.2, 0.1]]

        self.canvas.ax.plot(*initial_image)
        self._last_recip_update = 0.0
        self._mpl_cursor = None
        self._connect_tick_callbacks_once()
        self.canvas.figure.tight_layout()
        self.canvas.draw_idle()

    def update_image(self, array):
        self.canvas.ax.plot(*array)
        self.canvas.draw_idle()

    def reciprocal_ticks(self, mn, mx, n=4, intervals=[1, 2, 5, 10, 20, 50, 100]):
        ticks = []
        if mn == mx:
            return np.array([])
        low = min(mn, mx)
        high = max(mn, mx)
        if high <= 0:
            return np.array([])
        if low <= 0:
            low = 1e-12
        for i in intervals:
            if i/low - i/high < n:
                continue
            else:
                tick = np.ceil(i/low) / i
                while tick > 1/high:
                    ticks.append(tick)
                    tick -= 1.0 / i
                break
        return np.array(sorted(ticks))

    def _update_reciprocal_y_ticks(self, ax=None):
        """Compute labels and set reciprocal-style y-ticks. Debounced to avoid frequent runs."""
        # debounce: allow at most ~6-7 updates per second
        now = time.time()
        if now - getattr(self, "_last_recip_update", 0.0) < 0.15:
            return
        self._last_recip_update = now

        if ax is None:
            ax = self.canvas.ax
        ymin, ymax = ax.get_ylim()
        if ymax <= 0:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"{y:g}"))
            self.canvas.draw_idle()
            return

        ticks = self.reciprocal_ticks(ymin, ymax)
        if ticks.size == 0:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f"1/{1/y:.3g}" if y != 0 else "0"))
        else:
            ax.set_yticks(1/ticks)
            labels = []
            for val in ticks:
                if val == 0:
                    labels.append("0")
                else:
                    denom = val
                    if abs(denom - round(denom)) < 1e-8:
                        labels.append(f"1/{int(round(denom))}")
                    else:
                        labels.append(f"1/{denom:g}")
            ax.set_yticklabels(labels)

        # draw once after updates (debounced)
        self.canvas.draw_idle()

    def _connect_tick_callbacks_once(self):
        ax = self.canvas.ax
        if getattr(self, "_recip_callbacks_connected", False):
            return
        # connect to ylim_changed using a lambda that calls debounced updater
        ax.callbacks.connect('ylim_changed', lambda a: self._update_reciprocal_y_ticks(a))
        # also connect to draw_event as a fallback
        self.canvas.mpl_connect('draw_event', lambda event: self._update_reciprocal_y_ticks())
        self._recip_callbacks_connected = True

    def plot_score_vs_resolution(self, x, y, names=None, xlabel=None, ylabel=None):
        ax = self.canvas.ax
        ax.clear()

        sc = ax.scatter(x, y, s=10)

        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        # Clean up previous cursor if present
        try:
            if self._mpl_cursor is not None:
                self._mpl_cursor.disconnect()
        except Exception:
            pass

        # Reset active annotation holder
        self._active_annotation = None

        # Create an mplcursors cursor for this scatter collection
        self._mpl_cursor = mplcursors.cursor(sc, hover=True)

        def _on_add(sel):
            i = sel.index
            if names is not None:
                try:
                    ch, resatom = names[i].split(':',1)
                    label = f"Chain {ch[1:]}\nResidue {resatom.split('@',1)[0]}\nAtom {resatom.split('@',1)[1]}"
                except Exception:
                    label = str(i)
            else:
                label = str(i)

            # set text and style, and ensure visible
            sel.annotation.set_text(label)
            sel.annotation.get_bbox_patch().set_alpha(0.85)
            sel.annotation.set_visible(True)

            # hide any previously shown annotation (so old labels don't linger)
            if getattr(self, "_active_annotation", None) is not None and self._active_annotation is not sel.annotation:
                try:
                    self._active_annotation.set_visible(False)
                except Exception:
                    pass

            self._active_annotation = sel.annotation
            self.canvas.draw_idle()

        def _on_remove(sel):
            try:
                sel.annotation.set_visible(False)
            except Exception:
                pass

            if getattr(self, "_active_annotation", None) is sel.annotation:
                self._active_annotation = None

            self.canvas.draw_idle()

        _last_motion = {"t": 0.0}
        def _on_motion(event):
            # Only do work if an annotation is currently visible (common case: most motion is ignored)
            if getattr(self, "_active_annotation", None) is None:
                return

            # Throttle checks to reduce CPU (max ~10 fps here)
            now = time.time()
            if now - _last_motion["t"] < 0.1: 
                return
            _last_motion["t"] = now

            if event.inaxes != self.canvas.ax:
                # Moved outside axes — hide active annotation
                try:
                    if self._active_annotation is not None:
                        self._active_annotation.set_visible(False)
                        self._active_annotation = None
                        self.canvas.draw_idle()
                except Exception:
                    pass
                return

            # Check whether the mouse is over any scatter point
            try:
                contains, info = sc.contains(event)
            except Exception:
                contains = False

            if not contains:
                try:
                    if self._active_annotation is not None:
                        self._active_annotation.set_visible(False)
                        self._active_annotation = None
                        self.canvas.draw_idle()
                except Exception:
                    pass

        self._motion_cid = self.canvas.mpl_connect('motion_notify_event', _on_motion)
        self._mpl_cursor.connect('add', _on_add)
        self._mpl_cursor.connect('remove', _on_remove)
        self._update_reciprocal_y_ticks()
        self._connect_tick_callbacks_once()
        ax.figure.tight_layout()
        self.canvas.draw_idle()


class ScriptWorker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(object)   # can emit results
    error = pyqtSignal(str)
    ask = pyqtSignal(str)
    answer = pyqtSignal(bool)
    progress = pyqtSignal(str)

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
            self.progress.emit('Preprocessing complete')
            result = exposure(pdb_path=pre_out, out_path=out_path, progress_callback=self.progress.emit)
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
            self.progress.emit('Preprocessing complete')
            features_out = features(pdb_path=pre_out, feature=feature)
            result = pre_out, features_out
            self.finished.emit(result)
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
            result = exposure(pdb_path=pdb_path, out_path=out_path, assignment=assignment, progress_callback=self.progress.emit)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def run_plot(self):
        self.started.emit()
        try:
            settings = self.settings
            pdb_path = settings.get("pdb_path")
            defattr_path = settings.get("defattr_path")
            only_chain = settings.get("only_chain")

            result = score_v_localres(pdb_path=pdb_path, defattr_path=defattr_path, only_chain=only_chain, called_by_GUI=True, inverse=True)
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
        loop.exec()
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
        file_menu = self.menuBar().addMenu("&File")
        close_action = QAction("Close", self)
        close_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Close))             # maps to ⌘W on mac
        close_action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)  # optional: keep it global
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        self.addAction(close_action)

        # Set up Tabs
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
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
        self.file_browse = QPushButton("Browse...")
        self.file_browse.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_edit)
        file_row.addWidget(self.file_browse)
        simple_form.addRow("PDB File:", file_row)

        # Folder (preproccessed) selection
        folder_pre_row = QHBoxLayout()
        self.folder_pre_edit = QLineEdit()
        self.folder_pre_edit.setText(self.current_simple_settings.get("folder_pre_path", ""))
        self.folder_pre_browse = QPushButton("Browse...")
        self.folder_pre_browse.clicked.connect(self._browse_pre_folder)
        folder_pre_row.addWidget(self.folder_pre_edit)
        folder_pre_row.addWidget(self.folder_pre_browse)
        simple_form.addRow("Preprocessed Folder:", folder_pre_row)

        # Folder (out) selection
        folder_out_row = QHBoxLayout()
        self.folder_out_edit = QLineEdit()
        self.folder_out_edit.setText(self.current_simple_settings.get("folder_out_path", ""))
        self.folder_out_browse = QPushButton("Browse...")
        self.folder_out_browse.clicked.connect(self._browse_out_folder)
        folder_out_row.addWidget(self.folder_out_edit)
        folder_out_row.addWidget(self.folder_out_browse)
        simple_form.addRow("Output Folder:", folder_out_row)
        
        # Output text box
        self.simple_output = QTextEdit()
        self.simple_output.setReadOnly(True)
        self.simple_output.setPlaceholderText("Results will appear here...")
        simple_form.addRow('', self.simple_output)

        # Bottom: Run Button
        self.run_simple = QPushButton("Calculate")
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
        self.adduct_file_browse = QPushButton("Browse...")
        self.adduct_file_browse.clicked.connect(self._browse_adduct_file)
        adduct_file_row.addWidget(self.adduct_file_edit)
        adduct_file_row.addWidget(self.adduct_file_browse)
        adduct_form.addRow("PDB File:", adduct_file_row)

        # Folder (preprocessed) selection
        adduct_folder_pre_row = QHBoxLayout()
        self.adduct_folder_pre_edit = QLineEdit()
        self.adduct_folder_pre_edit.setText(self.current_adduct_settings.get("folder_pre_path", ""))
        self.adduct_folder_pre_browse = QPushButton("Browse...")
        self.adduct_folder_pre_browse.clicked.connect(self._browse_adduct_pre_folder)
        adduct_folder_pre_row.addWidget(self.adduct_folder_pre_edit)
        adduct_folder_pre_row.addWidget(self.adduct_folder_pre_browse)
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
        self.adduct_folder_out_browse = QPushButton("Browse...")
        self.adduct_folder_out_browse.clicked.connect(self._browse_adduct_out_folder)
        adduct_folder_out_row.addWidget(self.adduct_folder_out_edit)
        adduct_folder_out_row.addWidget(self.adduct_folder_out_browse)
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
        self.plot_pdb_browse = QPushButton("Browse...")
        self.plot_pdb_browse.clicked.connect(self._browse_plot_file)
        plot_pdb_row.addWidget(self.plot_pdb_edit)
        plot_pdb_row.addWidget(self.plot_pdb_browse)
        plot_form.addRow("PDB File:", plot_pdb_row)

        plot_defattr_row = QHBoxLayout()
        self.plot_defattr_edit = QLineEdit()
        self.plot_defattr_edit.setText(self.current_plot_settings.get("defattr_path", ""))
        self.plot_defattr_browse = QPushButton("Browse...")
        self.plot_defattr_browse.clicked.connect(self._browse_defattr_file)
        plot_defattr_row.addWidget(self.plot_defattr_edit)
        plot_defattr_row.addWidget(self.plot_defattr_browse)
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
        bottom_h.addWidget(self.interactive_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)

        # Plot button
        self.run_plot = QPushButton("Plot")
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
        updated_settings = self.get_simple_settings()
        for k in updated_settings:
            self.current_simple_settings[k] = updated_settings.get(k)

        # disable buttons while running
        self.run_simple.setEnabled(False)
        self.file_browse.setEnabled(False)
        self.folder_pre_browse.setEnabled(False)
        self.folder_out_browse.setEnabled(False)
        self.run_adduct_out.setEnabled(False)
        self.run_adduct_pre.setEnabled(False)
        self.run_plot.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_simple_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_simple)
        self.worker.progress.connect(self.simple_output.append)
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
        self.file_browse.setEnabled(True)
        self.folder_pre_browse.setEnabled(True)
        self.folder_out_browse.setEnabled(True)
        self.run_adduct_out.setEnabled(True)
        self.run_adduct_pre.setEnabled(True)
        self.run_plot.setEnabled(True)
        for i in result:
            self.simple_output.append(f"File {i[0]} saved. \n Min: {i[1]:2g} \n Max: {i[2]:2g}")

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
        self.run_simple.setEnabled(False)
        self.run_plot.setEnabled(False)
        self.run_adduct_out.setEnabled(False)
        self.run_adduct_pre.setEnabled(False)
        self.adduct_file_browse.setEnabled(False)
        self.adduct_folder_pre_browse.setEnabled(False)
        self.adduct_folder_out_browse.setEnabled(False)
        self.run_plot.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_adduct_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_adduct_pre)
        self.worker.progress.connect(self.adduct_output.append)
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
        self.run_simple.setEnabled(True)
        self.run_plot.setEnabled(True)
        self.run_adduct_out.setEnabled(True)
        self.run_adduct_pre.setEnabled(True)
        self.adduct_file_browse.setEnabled(True)
        self.adduct_folder_pre_browse.setEnabled(True)
        self.adduct_folder_out_browse.setEnabled(True)
        self.run_plot.setEnabled(True)     
        pre_out, options = result
        self.current_adduct_settings["pre_out_path"] = pre_out
        self.adduct_output.append(f"File {pre_out} saved. \n There were {len(options)} unique entries under {self.current_adduct_settings.get("feature")}.")
        self.combo.clear()
        self.combo.addItems(options)

    def on_worker_adduct_pre_error(self, err_str):
        self.run_adduct_pre.setEnabled(True)
        QMessageBox.critical(self, "Script error", f"An error occurred:\n{err_str}")

    def on_run_adduct_out_clicked(self):
        # gather values
        updated_settings = self.get_adduct_settings()
        for k in updated_settings:
            self.current_adduct_settings[k] = updated_settings.get(k)

        # disable run button while running
        self.run_simple.setEnabled(False)
        self.run_plot.setEnabled(False)
        self.run_adduct_out.setEnabled(False)
        self.run_adduct_pre.setEnabled(False)
        self.adduct_file_browse.setEnabled(False)
        self.adduct_folder_pre_browse.setEnabled(False)
        self.adduct_folder_out_browse.setEnabled(False)
        self.run_plot.setEnabled(False)

        # create worker & thread
        self.thread = QThread()
        self.worker = ScriptWorker(settings=self.current_adduct_settings.copy())
        self.worker.moveToThread(self.thread)

        # connect signals
        self.thread.started.connect(self.worker.run_adduct_out)
        self.worker.progress.connect(self.adduct_output.append)
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
        self.run_simple.setEnabled(True)
        self.run_plot.setEnabled(True)
        self.run_adduct_out.setEnabled(True)
        self.run_adduct_pre.setEnabled(True)
        self.adduct_file_browse.setEnabled(True)
        self.adduct_folder_pre_browse.setEnabled(True)
        self.adduct_folder_out_browse.setEnabled(True)    
        self.run_plot.setEnabled(True)
        for i in result:
            self.adduct_output.append(f"File {i[0]} saved. \n Min: {i[1]:2g} \n Max: {i[2]:2g}")

    def on_worker_adduct_out_error(self, err_str):
        self.run_adduct_out.setEnabled(True)
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
        fname, _ = QFileDialog.getOpenFileName(self, "Select file", self.file_edit.text() or "", "All Files (*)")
        if fname:
            self.file_edit.setText(fname)

    def _browse_adduct_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select file", self.adduct_file_edit.text() or "", "All Files (*)")
        if fname:
            self.adduct_file_edit.setText(fname)

    def _browse_plot_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select file", self.plot_pdb_edit.text() or "", "All Files (*)")
        if fname:
            self.plot_pdb_edit.setText(fname)

    def _browse_defattr_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select file", self.plot_defattr_edit.text() or "", "All Files (*)")
        if fname:
            self.plot_defattr_edit.setText(fname)

    def _browse_pre_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder", self.folder_pre_edit.text() or "")
        if folder:
            self.folder_pre_edit.setText(folder)
            
    def _browse_out_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder", self.folder_out_edit.text() or "")
        if folder:
            self.folder_out_edit.setText(folder)    
    
    def _browse_adduct_pre_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder", self.adduct_folder_pre_edit.text() or "")
        if folder:
            self.adduct_folder_pre_edit.setText(folder)
            
    def _browse_adduct_out_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder", self.adduct_folder_out_edit.text() or "")
        if folder:
            self.adduct_folder_out_edit.setText(folder)

    def _on_worker_ask_question(self, text):
        # This runs on the main thread (slot invoked in main thread)
        dlg = QMessageBox(self)
        dlg.setWindowTitle("User Input Required")
        dlg.setText(text)
        dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dlg.setDefaultButton(QMessageBox.StandardButton.Yes)
        dlg.setIcon(QMessageBox.Icon.Question)
        button = dlg.exec()
        yn = (button == QMessageBox.StandardButton.Yes)
        # send the answer back to the worker
        # worker.answer is a signal defined in worker; safe to emit from main thread
        self.worker.answer.emit(yn)

    def get_simple_settings(self):
        # Return a dict of settings
        return {
            "pdb_path": self.file_edit.text(),
            "folder_pre_path": self.folder_pre_edit.text(),
            "folder_out_path": self.folder_out_edit.text(),
        }
    
    def get_adduct_settings(self):
        return {
            "pdb_path": self.adduct_file_edit.text(),
            "folder_pre_path": self.adduct_folder_pre_edit.text(),
            "folder_out_path": self.adduct_folder_out_edit.text(),
            "feature": self.adduct_feature.currentText(),
            "combo": self.combo.currentData(),
        }
    
    def get_plot_settings(self):
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