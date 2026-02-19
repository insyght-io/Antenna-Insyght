# Antenna-Insyght — Dual-Band PCB IFA Antenna Design Tool
# Copyright (C) 2026 Insyght B.V.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""Main window for the Antenna Insyght GUI."""

import os
import sys

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QToolBar, QStatusBar, QLabel,
    QFileDialog, QMessageBox, QProgressBar,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QKeySequence

from gui.state import DesignState
from gui.panels.parameters import ParameterPanel
from gui.panels.geometry_view import GeometryView
from gui.panels.s11_view import S11View
from gui.panels.smith_view import SmithView
from gui.panels.impedance_view import ImpedanceView
from gui.panels.matching import MatchingPanel
from gui.workers.simulation import SimulationWorker
from gui.workers.matching_opt import MatchingOptWorker
from gui.dialogs.freq_bands import FreqBandsDialog
from gui.dialogs.sim_settings import SimSettingsDialog
from gui.dialogs.export_dialog import ExportDialog

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Antenna Insyght')
        self.resize(1400, 900)

        self.state = DesignState()
        self._sim_worker = None
        self._match_worker = None

        # FDTD settings (defaults)
        self._sim_timesteps = 80000
        self._sim_convergence = 1e-4
        self._sim_mesh_res = 20

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        # --- Menu Bar ---
        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        self._add_action(file_menu, '&New', self._on_new, QKeySequence.New)
        self._add_action(file_menu, '&Open JSON...', self._on_open, QKeySequence.Open)
        self._add_action(file_menu, '&Save JSON...', self._on_save, QKeySequence.Save)
        file_menu.addSeparator()
        self._add_action(file_menu, 'Save &NPZ...', self._on_save_npz)
        self._add_action(file_menu, 'Export &DXF...', self._on_export_dxf)
        self._add_action(file_menu, 'Export &KiCad...', self._on_export_kicad)
        self._add_action(file_menu, 'Generate &PDF...', self._on_generate_pdf)
        file_menu.addSeparator()
        self._add_action(file_menu, '&Quit', self.close, QKeySequence.Quit)

        sim_menu = menubar.addMenu('&Simulate')
        self._add_action(sim_menu, 'Quick &Explore (30k)', lambda: self._run_sim('explore'))
        self._add_action(sim_menu, '&Standard (80k)', lambda: self._run_sim('standard'))
        self._add_action(sim_menu, '&Full Accuracy (200k)', lambda: self._run_sim('full'))
        sim_menu.addSeparator()
        self._act_cancel = self._add_action(sim_menu, '&Cancel', self._on_cancel_sim)
        self._act_cancel.setEnabled(False)
        sim_menu.addSeparator()
        self._add_action(sim_menu, 'S&ettings...', self._on_sim_settings)

        opt_menu = menubar.addMenu('&Optimize')
        self._add_action(opt_menu, '&Matching Network Only', self._on_optimize_matching)
        self._add_action(opt_menu, '&Round to E24', self._on_round_e24)
        opt_menu.addSeparator()
        self._add_action(opt_menu, '&Frequency Bands...', self._on_freq_bands)

        help_menu = menubar.addMenu('&Help')
        self._add_action(help_menu, '&About', self._on_about)
        self._add_action(help_menu, '&LTE Band Reference', self._on_band_ref)

        # --- Toolbar ---
        toolbar = QToolBar('Main')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addAction('New').triggered.connect(self._on_new)
        toolbar.addAction('Open').triggered.connect(self._on_open)
        toolbar.addAction('Save').triggered.connect(self._on_save)
        toolbar.addSeparator()
        self._btn_explore = toolbar.addAction('Explore')
        self._btn_explore.triggered.connect(lambda: self._run_sim('explore'))
        self._btn_simulate = toolbar.addAction('Simulate')
        self._btn_simulate.triggered.connect(lambda: self._run_sim('standard'))
        self._btn_full = toolbar.addAction('Full')
        self._btn_full.triggered.connect(lambda: self._run_sim('full'))

        # --- Left Dock: Parameters ---
        self.params_panel = ParameterPanel(self.state, self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.params_panel)

        # --- Central Tabs ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.geometry_view = GeometryView(self.state, self)
        self.s11_view = S11View(self.state, self)
        self.smith_view = SmithView(self.state, self)
        self.matching_panel = MatchingPanel(self.state, self)
        self.impedance_view = ImpedanceView(self.state, self)

        self.tabs.addTab(self.geometry_view, 'Geometry')
        self.tabs.addTab(self.s11_view, 'S11')
        self.tabs.addTab(self.smith_view, 'Smith')
        self.tabs.addTab(self.matching_panel, 'Matching')
        self.tabs.addTab(self.impedance_view, 'Impedance')

        # --- Status Bar ---
        self.statusBar().showMessage('Ready')
        self._lbl_sim = QLabel('Sim: idle')
        self._lbl_lb = QLabel('LB: --')
        self._lbl_hb = QLabel('HB: --')
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(150)
        self._progress.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().addPermanentWidget(self._lbl_sim)
        self.statusBar().addPermanentWidget(self._lbl_lb)
        self.statusBar().addPermanentWidget(self._lbl_hb)

    def _add_action(self, menu, text, slot, shortcut=None):
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _connect_signals(self):
        self.params_panel.simulate_clicked.connect(lambda: self._run_sim('standard'))
        self.matching_panel.optimize_clicked.connect(self._on_optimize_matching_topology)
        self.state.simulation_finished.connect(self._on_sim_result)

    # ---- Simulation ----

    @Slot()
    def _run_sim(self, mode):
        if self._sim_worker and self._sim_worker.isRunning():
            self.statusBar().showMessage('Simulation already running')
            return

        self._lbl_sim.setText(f'Sim: {mode}...')
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # indeterminate
        self._act_cancel.setEnabled(True)
        self._set_sim_buttons(False)

        self._sim_worker = SimulationWorker(self.state.antenna_params_dict(), mode)
        self._sim_worker.finished.connect(self._on_sim_finished)
        self._sim_worker.error.connect(self._on_sim_error)
        self._sim_worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self._sim_worker.start()

    @Slot(dict)
    def _on_sim_finished(self, result):
        self.state.sim_result = result
        self.state.simulation_finished.emit(result)
        self._progress.setVisible(False)
        self._act_cancel.setEnabled(False)
        self._set_sim_buttons(True)

    @Slot(dict)
    def _on_sim_result(self, result):
        import numpy as np
        freqs = result['freqs']
        s11_db = result['s11_db']
        lb_mask = (freqs >= self.state.freq_bands['LB_MIN']) & (freqs <= self.state.freq_bands['LB_MAX'])
        hb_mask = (freqs >= self.state.freq_bands['HB_MIN']) & (freqs <= self.state.freq_bands['HB_MAX'])

        lb_worst = float(np.max(s11_db[lb_mask])) if np.any(lb_mask) else float('nan')
        hb_worst = float(np.max(s11_db[hb_mask])) if np.any(hb_mask) else float('nan')

        self._lbl_sim.setText('Sim: done')
        self._lbl_lb.setText(f'LB: {lb_worst:.1f} dB')
        self._lbl_hb.setText(f'HB: {hb_worst:.1f} dB')
        self.statusBar().showMessage(f'Simulation complete. LB={lb_worst:.1f} dB, HB={hb_worst:.1f} dB')

    @Slot(str)
    def _on_sim_error(self, msg):
        self._lbl_sim.setText('Sim: error')
        self._progress.setVisible(False)
        self._act_cancel.setEnabled(False)
        self._set_sim_buttons(True)
        QMessageBox.warning(self, 'Simulation Error', msg)

    @Slot()
    def _on_cancel_sim(self):
        if self._sim_worker and self._sim_worker.isRunning():
            self._sim_worker.terminate()
            self._lbl_sim.setText('Sim: cancelled')
            self._progress.setVisible(False)
            self._act_cancel.setEnabled(False)
            self._set_sim_buttons(True)

    def _set_sim_buttons(self, enabled):
        self._btn_explore.setEnabled(enabled)
        self._btn_simulate.setEnabled(enabled)
        self._btn_full.setEnabled(enabled)

    # ---- Matching ----

    @Slot()
    def _on_optimize_matching(self):
        topology = self.matching_panel.current_topology()
        if topology:
            self._on_optimize_matching_topology(topology)

    @Slot(str)
    def _on_optimize_matching_topology(self, topology_name):
        if self.state.sim_result is None:
            QMessageBox.information(self, 'No Data', 'Run a simulation first to get Z11 data.')
            return
        if self._match_worker and self._match_worker.isRunning():
            self.statusBar().showMessage('Matching optimization already running')
            return

        self.statusBar().showMessage(f'Optimizing {topology_name}...')
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)

        self._match_worker = MatchingOptWorker(
            topology_name,
            self.state.sim_result['freqs'],
            self.state.sim_result['z_complex'],
            n_runs=10,
        )
        self._match_worker.finished.connect(self._on_match_finished)
        self._match_worker.error.connect(self._on_match_error)
        self._match_worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self._match_worker.start()

    @Slot(dict)
    def _on_match_finished(self, result):
        self._progress.setVisible(False)
        self.state.matching_topology = result.get('topology')
        self.state.matching_values = result.get('values')
        self.state.matching_result = result
        self.state.matching_finished.emit(result)
        lb = result.get('lb_worst', float('nan'))
        hb = result.get('hb_worst', float('nan'))
        self.statusBar().showMessage(
            f'Matching done: LB={lb:.1f} dB, HB={hb:.1f} dB')

    @Slot(str)
    def _on_match_error(self, msg):
        self._progress.setVisible(False)
        QMessageBox.warning(self, 'Matching Error', msg)

    @Slot()
    def _on_round_e24(self):
        self.matching_panel._on_round_e24()

    # ---- File Operations ----

    @Slot()
    def _on_new(self):
        self.state.params = dict(DesignState.DEFAULT_PARAMS)
        self.state.freq_bands = dict(DesignState.DEFAULT_FREQ)
        self.state.sim_result = None
        self.state.matching_topology = None
        self.state.matching_values = None
        self.state.matching_result = None
        self.state.params_changed.emit()
        self.state.freq_bands_changed.emit()
        self._lbl_sim.setText('Sim: idle')
        self._lbl_lb.setText('LB: --')
        self._lbl_hb.setText('HB: --')
        self.statusBar().showMessage('New design')

    @Slot()
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Design', '', 'JSON Files (*.json)')
        if path:
            try:
                self.state.load_json(path)
                self.statusBar().showMessage(f'Loaded: {path}')
            except Exception as e:
                QMessageBox.warning(self, 'Load Error', str(e))

    @Slot()
    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Design', 'antenna_design.json',
                                               'JSON Files (*.json)')
        if path:
            try:
                self.state.save_json(path)
                self.statusBar().showMessage(f'Saved: {path}')
            except Exception as e:
                QMessageBox.warning(self, 'Save Error', str(e))

    @Slot()
    def _on_save_npz(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save NPZ', 'antenna_data.npz',
                                               'NumPy Files (*.npz)')
        if path:
            try:
                self.state.save_npz(path)
                self.statusBar().showMessage(f'Saved: {path}')
            except Exception as e:
                QMessageBox.warning(self, 'Save Error', str(e))

    @Slot()
    def _on_export_dxf(self):
        dlg = ExportDialog(parent=self)
        if dlg.exec():
            result = dlg.result_options()
            try:
                sys.path.insert(0, SCRIPT_DIR)
                from export_dxf import export_branched_ifa_dxf
                from antenna_model import BranchedIFA
                ant = BranchedIFA(self.state.antenna_params_dict())
                out = os.path.join(result['output_dir'], 'antenna.dxf')
                export_branched_ifa_dxf(ant, out)
                self.statusBar().showMessage(f'DXF exported: {out}')
            except Exception as e:
                QMessageBox.warning(self, 'Export Error', str(e))

    @Slot()
    def _on_export_kicad(self):
        dlg = ExportDialog(parent=self)
        if dlg.exec():
            result = dlg.result_options()
            try:
                sys.path.insert(0, SCRIPT_DIR)
                from create_kicad_project import main as kicad_main
                kicad_main()
                self.statusBar().showMessage('KiCad project generated')
            except Exception as e:
                QMessageBox.warning(self, 'Export Error', str(e))

    @Slot()
    def _on_generate_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save PDF Report', 'antenna_report.pdf',
                                               'PDF Files (*.pdf)')
        if path:
            try:
                sys.path.insert(0, SCRIPT_DIR)
                from report import generate_report_branched
                generate_report_branched(output_path=path)
                self.statusBar().showMessage(f'PDF generated: {path}')
            except Exception as e:
                QMessageBox.warning(self, 'PDF Error', str(e))

    # ---- Dialogs ----

    @Slot()
    def _on_freq_bands(self):
        dlg = FreqBandsDialog(self.state, self)
        dlg.exec()

    @Slot()
    def _on_sim_settings(self):
        dlg = SimSettingsDialog(
            current_settings={
                'max_timesteps': self._sim_timesteps,
                'end_criteria': self._sim_convergence,
                'mesh_res': self._sim_mesh_res,
            },
            parent=self,
        )
        if dlg.exec():
            s = dlg.settings()
            self._sim_timesteps = s['max_timesteps']
            self._sim_convergence = s['end_criteria']
            self._sim_mesh_res = s['mesh_res']
            self.statusBar().showMessage(
                f'Sim settings: {self._sim_timesteps} steps, {self._sim_convergence:.0e} convergence')

    @Slot()
    def _on_about(self):
        QMessageBox.about(self, 'About Antenna Insyght',
                          'Antenna Insyght v2.0\n\n'
                          'Branched IFA antenna design tool with openEMS FDTD simulation.\n\n'
                          'Design B: R=31 groot board, 8-element LCLCLCLC matching.\n'
                          'Targets: LB 791-960 MHz, HB 1710-1990 MHz.')

    @Slot()
    def _on_band_ref(self):
        QMessageBox.information(self, 'LTE Band Reference',
                                'Low Band:\n'
                                '  B20: 791-862 MHz (EU)\n'
                                '  B8:  880-960 MHz (EU)\n'
                                '  B12: 699-746 MHz (US)\n'
                                '  B13: 746-787 MHz (US)\n\n'
                                'High Band:\n'
                                '  B3:  1710-1880 MHz (EU)\n'
                                '  B2:  1850-1990 MHz (US)\n'
                                '  B4:  1710-2155 MHz (US)')
