#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

COMMAND_TOPIC = "/agv/rpm_cmd"
CONTROL_MODE_TOPIC = "/agv/control_mode"
CONTROL_STATUS_TOPIC = "/agv/position_control_status"
SEARCHING_MODE_CMD_TOPIC = "/searching_mode/cmd"
SEARCHING_MODE_STATUS_TOPIC = "/searching_mode/status"
RADAR_IMAGE_TOPIC = "/agv/lidar_radar_image"
CONTROL_MODE_MANUAL = "manual"
CONTROL_MODE_AUTOMATIC = "automatic"
DRIVE_DIRECTION_FORWARD = "forward"
FREE_MODE_COMMAND = "s"
MAX_WHEEL_SPEED = 150
MIN_FORWARD_SPEED = 50
MAX_SPEED_PERCENT = 100
PUBLISH_PERIOD_SEC = 0.10
COMMAND_TIMEOUT_SEC = 0.50
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
UBUNTU_HOTSPOT_HOST = "10.42.0.1"
INVERT_TURN_DIRECTION = False
INDICATOR_ON_COLOR = "#2e7d32"
INDICATOR_OFF_COLOR = "#9e9e9e"
SELECTED_TAB_SPEED_TURN = "speed_turn"
SELECTED_TAB_INDEPENDENT = "independent"


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AGV Control</title>
  <style>
    :root {
      --tk-bg: #f0f0f0;
      --tk-border: #b8b8b8;
      --tk-dark-border: #8f8f8f;
      --text: #111111;
      --manual: #2e7d32;
      --manual-active: #1b5e20;
      --automatic: #1565c0;
      --automatic-active: #0d47a1;
      --stop: #c62828;
      --stop-active: #8e0000;
      --cobot: #6a1b9a;
      --cobot-active: #4a148c;
      --indicator-off: #9e9e9e;
      --indicator-on: #2e7d32;
      --indicator-refining: #f9a825;
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      margin: 0;
      min-height: 100%;
      background: var(--tk-bg);
      color: var(--text);
      font-family: TkDefaultFont, "Segoe UI", Arial, sans-serif;
      font-size: 15px;
      letter-spacing: 0;
    }

    body {
      display: flex;
      justify-content: center;
      align-items: flex-start;
      padding: 12px;
      touch-action: manipulation;
    }

    .panel {
      width: min(100%, 386px);
      padding: 12px;
      background: var(--tk-bg);
    }

    .mode-button,
    .stop-button,
    .cobot-button {
      width: 100%;
      min-height: 48px;
      border: 2px solid var(--tk-dark-border);
      color: #ffffff;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      -webkit-tap-highlight-color: transparent;
    }

    .mode-button.manual {
      background: var(--manual);
    }

    .mode-button.manual:active {
      background: var(--manual-active);
    }

    .mode-button.automatic {
      background: var(--automatic);
    }

    .mode-button.automatic:active {
      background: var(--automatic-active);
    }

    .status-row {
      display: grid;
      grid-template-columns: 18px minmax(0, 1fr) 18px minmax(0, 1fr) 18px minmax(0, 1fr);
      column-gap: 6px;
      align-items: center;
      margin-top: 12px;
    }

    .status-label {
      padding-right: 18px;
      white-space: nowrap;
    }

    .indicator {
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--indicator-off);
    }

    .indicator.on {
      background: var(--indicator-on);
    }

    .indicator.refining {
      background: var(--indicator-refining);
    }

    .notebook {
      margin-top: 12px;
    }

    .tabs {
      display: flex;
      align-items: flex-end;
    }

    .tab {
      min-height: 30px;
      padding: 4px 10px;
      border: 1px solid var(--tk-border);
      border-bottom: 0;
      background: #e7e7e7;
      color: var(--text);
      font: inherit;
      cursor: pointer;
    }

    .tab.active {
      position: relative;
      top: 1px;
      background: var(--tk-bg);
      border-color: var(--tk-dark-border);
      z-index: 1;
    }

    .tab-panel {
      display: none;
      min-height: 118px;
      padding: 12px;
      border: 1px solid var(--tk-dark-border);
      background: var(--tk-bg);
    }

    .tab-panel.active {
      display: block;
    }

    .slider-row {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr) 38px;
      column-gap: 8px;
      align-items: center;
      margin-bottom: 12px;
    }

    .slider-row:last-child {
      margin-bottom: 0;
    }

    .slider-row label {
      white-space: nowrap;
    }

    .slider-value {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    input[type="range"] {
      width: 100%;
      min-width: 0;
      margin: 0;
    }

    .command-label {
      margin-top: 12px;
      margin-bottom: 8px;
      min-height: 22px;
      font-size: 15px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    .stop-button {
      background: var(--stop);
    }

    .stop-button:active {
      background: var(--stop-active);
    }

    .cobot-button {
      margin-top: 8px;
      background: var(--cobot);
    }

    .cobot-button:active {
      background: var(--cobot-active);
    }

    .cobot-button.running {
      background: var(--stop);
    }

    .cobot-button.running:active {
      background: var(--stop-active);
    }

    .radar-section {
      margin-top: 12px;
    }

    .radar-header {
      margin-bottom: 6px;
      font-size: 15px;
      font-weight: 700;
    }

    .radar-frame {
      width: 100%;
      aspect-ratio: 1 / 1;
      border: 1px solid var(--tk-dark-border);
      background: #000000;
      display: grid;
      place-items: center;
      overflow: hidden;
    }

    .radar-frame img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .radar-status {
      color: #ffffff;
      font-weight: 600;
      text-align: center;
      padding: 12px;
    }

    @media (max-width: 360px) {
      body {
        padding: 8px;
      }

      .panel {
        padding: 10px;
      }

      .slider-row {
        grid-template-columns: 86px minmax(0, 1fr) 34px;
        column-gap: 6px;
      }

      .tab {
        padding-left: 8px;
        padding-right: 8px;
      }
    }
  </style>
</head>
<body>
  <main class="panel" aria-label="AGV Control">
    <button id="modeButton" class="mode-button manual" type="button">
      Mode: MANUAL
    </button>

    <section class="status-row" aria-label="Control status">
      <span id="orientationIndicator" class="indicator"></span>
      <span class="status-label">Orientation</span>
      <span id="positionIndicator" class="indicator"></span>
      <span class="status-label">Position</span>
      <span id="harvestingIndicator" class="indicator"></span>
      <span class="status-label">Harvesting</span>
    </section>

    <section class="notebook">
      <div class="tabs" role="tablist">
        <button
          id="speedTurnTab"
          class="tab active"
          type="button"
          role="tab"
          aria-controls="speedTurnPanel"
        >Speed + Turn</button>
        <button
          id="independentTab"
          class="tab"
          type="button"
          role="tab"
          aria-controls="independentPanel"
        >Independent wheels</button>
      </div>

      <div id="speedTurnPanel" class="tab-panel active" role="tabpanel">
        <div class="slider-row">
          <label for="speedSlider">Speed (%)</label>
          <input id="speedSlider" type="range" min="0" max="100" step="1" value="0">
          <span id="speedValue" class="slider-value">0</span>
        </div>
        <div class="slider-row">
          <label for="turnSlider">Turn</label>
          <input id="turnSlider" type="range" min="-100" max="100" step="1" value="0">
          <span id="turnValue" class="slider-value">0</span>
        </div>
      </div>

      <div id="independentPanel" class="tab-panel" role="tabpanel">
        <div class="slider-row">
          <label for="rightSlider">Right wheel</label>
          <input id="rightSlider" type="range" min="0" max="150" step="1" value="0">
          <span id="rightValue" class="slider-value">0</span>
        </div>
        <div class="slider-row">
          <label for="leftSlider">Left wheel</label>
          <input id="leftSlider" type="range" min="0" max="150" step="1" value="0">
          <span id="leftValue" class="slider-value">0</span>
        </div>
      </div>
    </section>

    <div id="commandLabel" class="command-label">
      Command: 0,0   right=0, left=0
    </div>

    <button id="stopButton" class="stop-button" type="button">STOP</button>
    <button id="cobotButton" class="cobot-button" type="button">Cobot</button>

    <section class="radar-section" aria-label="LiDAR radar">
      <div class="radar-header">LiDAR Radar</div>
      <div class="radar-frame">
        <img id="radarImage" alt="LiDAR radar" hidden>
        <div id="radarStatus" class="radar-status">Waiting for LiDAR radar</div>
      </div>
    </section>
  </main>

  <script>
    const SELECTED_TAB_SPEED_TURN = "speed_turn";
    const SELECTED_TAB_INDEPENDENT = "independent";
    const DRIVE_DIRECTION_FORWARD = "forward";

    const state = {
      selected_tab: SELECTED_TAB_SPEED_TURN,
      drive_direction: DRIVE_DIRECTION_FORWARD,
      speed_percent: 0,
      turn: 0,
      right_wheel: 0,
      left_wheel: 0
    };

    const elements = {
      modeButton: document.getElementById("modeButton"),
      stopButton: document.getElementById("stopButton"),
      cobotButton: document.getElementById("cobotButton"),
      orientationIndicator: document.getElementById("orientationIndicator"),
      positionIndicator: document.getElementById("positionIndicator"),
      harvestingIndicator: document.getElementById("harvestingIndicator"),
      speedTurnTab: document.getElementById("speedTurnTab"),
      independentTab: document.getElementById("independentTab"),
      speedTurnPanel: document.getElementById("speedTurnPanel"),
      independentPanel: document.getElementById("independentPanel"),
      speedSlider: document.getElementById("speedSlider"),
      turnSlider: document.getElementById("turnSlider"),
      rightSlider: document.getElementById("rightSlider"),
      leftSlider: document.getElementById("leftSlider"),
      speedValue: document.getElementById("speedValue"),
      turnValue: document.getElementById("turnValue"),
      rightValue: document.getElementById("rightValue"),
      leftValue: document.getElementById("leftValue"),
      commandLabel: document.getElementById("commandLabel"),
      radarImage: document.getElementById("radarImage"),
      radarStatus: document.getElementById("radarStatus")
    };

    let inFlight = false;
    let pendingAction = null;

    function numberFromSlider(slider) {
      return Number.parseInt(slider.value, 10);
    }

    function updateLocalSliderState() {
      state.speed_percent = numberFromSlider(elements.speedSlider);
      state.turn = numberFromSlider(elements.turnSlider);
      state.right_wheel = numberFromSlider(elements.rightSlider);
      state.left_wheel = numberFromSlider(elements.leftSlider);

      elements.speedValue.textContent = String(state.speed_percent);
      elements.turnValue.textContent = String(state.turn);
      elements.rightValue.textContent = String(state.right_wheel);
      elements.leftValue.textContent = String(state.left_wheel);
    }

    function renderTabs() {
      const independent = state.selected_tab === SELECTED_TAB_INDEPENDENT;
      elements.speedTurnTab.classList.toggle("active", !independent);
      elements.independentTab.classList.toggle("active", independent);
      elements.speedTurnPanel.classList.toggle("active", !independent);
      elements.independentPanel.classList.toggle("active", independent);
    }

    function applyServerState(serverState) {
      state.selected_tab = serverState.selected_tab;
      state.drive_direction = DRIVE_DIRECTION_FORWARD;
      state.speed_percent = serverState.speed_percent;
      state.turn = serverState.turn;
      state.right_wheel = serverState.right_wheel;
      state.left_wheel = serverState.left_wheel;

      elements.speedSlider.value = String(state.speed_percent);
      elements.turnSlider.value = String(state.turn);
      elements.rightSlider.value = String(state.right_wheel);
      elements.leftSlider.value = String(state.left_wheel);
      updateLocalSliderState();
      renderTabs();

      elements.modeButton.textContent = serverState.mode_label;
      elements.modeButton.classList.toggle(
        "automatic",
        serverState.control_mode === "automatic"
      );
      elements.modeButton.classList.toggle(
        "manual",
        serverState.control_mode !== "automatic"
      );
      elements.orientationIndicator.classList.toggle(
        "on",
        Boolean(serverState.orientation_control_active)
      );
      elements.positionIndicator.classList.toggle(
        "on",
        Boolean(serverState.position_control_active)
      );
      elements.harvestingIndicator.classList.toggle(
        "on",
        Boolean(serverState.harvesting_active)
      );
      elements.harvestingIndicator.classList.toggle(
        "refining",
        Boolean(serverState.harvesting_refining_active)
      );
      elements.commandLabel.textContent = serverState.command_label;
      elements.cobotButton.textContent = serverState.cobot_label;
      elements.cobotButton.classList.toggle(
        "running",
        Boolean(serverState.searching_mode_active)
      );
    }

    function payload(action) {
      updateLocalSliderState();
      return {
        action,
        selected_tab: state.selected_tab,
        drive_direction: state.drive_direction,
        speed_percent: state.speed_percent,
        turn: state.turn,
        right_wheel: state.right_wheel,
        left_wheel: state.left_wheel
      };
    }

    async function sendControl(action) {
      if (inFlight) {
        if (action !== "heartbeat") {
          pendingAction = action;
        }
        return;
      }

      inFlight = true;
      try {
        const response = await fetch("/api/control", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload(action)),
          cache: "no-store"
        });

        if (response.ok) {
          applyServerState(await response.json());
        }
      } catch (error) {
        // The ROS node deadman stops the AGV if this heartbeat disappears.
      } finally {
        inFlight = false;
        if (pendingAction !== null) {
          const nextAction = pendingAction;
          pendingAction = null;
          sendControl(nextAction);
        }
      }
    }

    function queueUpdate() {
      updateLocalSliderState();
      sendControl("update");
    }

    function selectTab(selectedTab) {
      state.selected_tab = selectedTab;
      renderTabs();
      sendControl("update");
    }

    function stopAgv() {
      state.selected_tab = SELECTED_TAB_SPEED_TURN;
      state.drive_direction = DRIVE_DIRECTION_FORWARD;
      elements.speedSlider.value = "0";
      elements.turnSlider.value = "0";
      elements.rightSlider.value = "0";
      elements.leftSlider.value = "0";
      updateLocalSliderState();
      renderTabs();
      sendControl("stop");
    }

    function stopWithBeacon() {
      const stopPayload = {
        action: "stop",
        selected_tab: SELECTED_TAB_SPEED_TURN,
        drive_direction: DRIVE_DIRECTION_FORWARD,
        speed_percent: 0,
        turn: 0,
        right_wheel: 0,
        left_wheel: 0
      };
      const data = new Blob(
        [JSON.stringify(stopPayload)],
        {type: "application/json"}
      );
      navigator.sendBeacon("/api/control", data);
    }

    function refreshRadarImage() {
      const nextImage = new Image();
      nextImage.onload = () => {
        elements.radarImage.src = nextImage.src;
        elements.radarImage.hidden = false;
        elements.radarStatus.hidden = true;
      };
      nextImage.onerror = () => {
        elements.radarImage.hidden = true;
        elements.radarStatus.hidden = false;
      };
      nextImage.src = `/api/radar.jpg?t=${Date.now()}`;
    }

    elements.modeButton.addEventListener(
      "click",
      () => sendControl("toggle_mode")
    );
    elements.stopButton.addEventListener("click", stopAgv);
    elements.cobotButton.addEventListener(
      "click",
      () => sendControl("cobot")
    );
    elements.speedTurnTab.addEventListener(
      "click",
      () => selectTab(SELECTED_TAB_SPEED_TURN)
    );
    elements.independentTab.addEventListener(
      "click",
      () => selectTab(SELECTED_TAB_INDEPENDENT)
    );

    elements.speedSlider.addEventListener("input", queueUpdate);
    elements.turnSlider.addEventListener("input", queueUpdate);
    elements.rightSlider.addEventListener("input", queueUpdate);
    elements.leftSlider.addEventListener("input", queueUpdate);
    window.addEventListener("pagehide", stopWithBeacon);

    renderTabs();
    updateLocalSliderState();
    sendControl("heartbeat");
    window.setInterval(() => sendControl("heartbeat"), 100);
    refreshRadarImage();
    window.setInterval(refreshRadarImage, 250);
  </script>
</body>
</html>
"""


class _AgvThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class AgvWebControlNode(Node):
    """Mobile web control panel for the AGV wheel command topic."""

    def __init__(self) -> None:
        super().__init__("agv_web_control_node")

        self.control_mode = CONTROL_MODE_MANUAL
        self.selected_tab = SELECTED_TAB_SPEED_TURN
        self.drive_direction = DRIVE_DIRECTION_FORWARD
        self.speed_percent = 0
        self.turn = 0
        self.right_wheel = 0
        self.left_wheel = 0
        self.position_control_active = False
        self.orientation_control_active = False
        self.harvesting_active = False
        self.harvesting_refining_active = False
        self.automatic_right_rpm = 0
        self.automatic_left_rpm = 0
        self.latest_radar_image: bytes | None = None
        self.latest_radar_image_time: float | None = None
        self.last_browser_command_time: float | None = None
        self.safety_stop_active = False
        self.free_stop_active = False
        self.searching_mode_active = False
        self.is_shutting_down = False
        self.lock = threading.RLock()

        self._declare_parameters()
        self._load_parameters()
        self._create_ros_interfaces()
        self._start_web_server()

        self.publish_timer = self.create_timer(
            self.publish_period_sec,
            self._on_publish_timer,
        )

        self.get_logger().info("agv_web_control_node ready")
        self.get_logger().info(f"Publishing to {self.command_topic}")
        self.get_logger().info(
            f"Publishing control mode to {self.control_mode_topic}"
        )
        self.get_logger().info(
            f"Publishing searching mode commands to {self.searching_mode_cmd_topic}"
        )
        self.get_logger().info(
            f"Subscribing to searching mode status on {self.searching_mode_status_topic}"
        )
        self.get_logger().info(
            f"Subscribing to control status on {self.control_status_topic}"
        )
        self.get_logger().info(
            f"Subscribing to radar image on {self.radar_image_topic}"
        )
        self.get_logger().info(
            f"Web control listening on http://{self.web_host}:{self.web_port}"
        )
        self._log_web_access_instructions()

    def _declare_parameters(self) -> None:
        self.declare_parameter("command_topic", COMMAND_TOPIC)
        self.declare_parameter("control_mode_topic", CONTROL_MODE_TOPIC)
        self.declare_parameter("control_status_topic", CONTROL_STATUS_TOPIC)
        self.declare_parameter("searching_mode_cmd_topic", SEARCHING_MODE_CMD_TOPIC)
        self.declare_parameter(
            "searching_mode_status_topic",
            SEARCHING_MODE_STATUS_TOPIC,
        )
        self.declare_parameter("radar_image_topic", RADAR_IMAGE_TOPIC)
        self.declare_parameter("invert_turn_direction", INVERT_TURN_DIRECTION)
        self.declare_parameter("web_host", WEB_HOST)
        self.declare_parameter("web_port", WEB_PORT)
        self.declare_parameter("publish_period_sec", PUBLISH_PERIOD_SEC)
        self.declare_parameter("command_timeout_sec", COMMAND_TIMEOUT_SEC)

    def _load_parameters(self) -> None:
        self.command_topic = str(self.get_parameter("command_topic").value).strip()
        if not self.command_topic:
            self.command_topic = COMMAND_TOPIC

        self.control_mode_topic = str(
            self.get_parameter("control_mode_topic").value
        ).strip()
        if not self.control_mode_topic:
            self.control_mode_topic = CONTROL_MODE_TOPIC

        self.control_status_topic = str(
            self.get_parameter("control_status_topic").value
        ).strip()
        if not self.control_status_topic:
            self.control_status_topic = CONTROL_STATUS_TOPIC

        self.searching_mode_cmd_topic = str(
            self.get_parameter("searching_mode_cmd_topic").value
        ).strip()
        if not self.searching_mode_cmd_topic:
            self.searching_mode_cmd_topic = SEARCHING_MODE_CMD_TOPIC

        self.searching_mode_status_topic = str(
            self.get_parameter("searching_mode_status_topic").value
        ).strip()
        if not self.searching_mode_status_topic:
            self.searching_mode_status_topic = SEARCHING_MODE_STATUS_TOPIC

        self.radar_image_topic = str(
            self.get_parameter("radar_image_topic").value
        ).strip()
        if not self.radar_image_topic:
            self.radar_image_topic = RADAR_IMAGE_TOPIC

        self.invert_turn_direction = bool(
            self.get_parameter("invert_turn_direction").value
        )

        self.web_host = str(self.get_parameter("web_host").value).strip()
        if not self.web_host:
            self.web_host = WEB_HOST

        self.web_port = self._int_parameter("web_port", WEB_PORT, 1, 65535)
        self.publish_period_sec = self._float_parameter(
            "publish_period_sec",
            PUBLISH_PERIOD_SEC,
            minimum=0.01,
        )
        self.command_timeout_sec = self._float_parameter(
            "command_timeout_sec",
            COMMAND_TIMEOUT_SEC,
            minimum=0.05,
        )

    def _int_parameter(
        self,
        name: str,
        default_value: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            value = int(self.get_parameter(name).value)
        except (TypeError, ValueError):
            self.get_logger().warn(
                f"Parameter '{name}' must be an integer; using {default_value}"
            )
            return default_value

        if value < minimum or value > maximum:
            self.get_logger().warn(
                "Parameter "
                f"'{name}' must be between {minimum} and {maximum}; "
                f"using {default_value}"
            )
            return default_value

        return value

    def _float_parameter(
        self,
        name: str,
        default_value: float,
        minimum: float,
    ) -> float:
        try:
            value = float(self.get_parameter(name).value)
        except (TypeError, ValueError):
            self.get_logger().warn(
                f"Parameter '{name}' must be numeric; using {default_value}"
            )
            return default_value

        if value < minimum:
            self.get_logger().warn(
                f"Parameter '{name}' must be >= {minimum}; using {default_value}"
            )
            return default_value

        return value

    def _create_ros_interfaces(self) -> None:
        self.pub_command = self.create_publisher(String, self.command_topic, 10)
        self.pub_control_mode = self.create_publisher(
            String,
            self.control_mode_topic,
            10,
        )
        self.pub_searching_mode_cmd = self.create_publisher(
            String,
            self.searching_mode_cmd_topic,
            10,
        )
        self.control_status_sub = self.create_subscription(
            String,
            self.control_status_topic,
            self._on_control_status_received,
            10,
        )
        self.searching_mode_status_sub = self.create_subscription(
            String,
            self.searching_mode_status_topic,
            self._on_searching_mode_status_received,
            10,
        )
        self.radar_image_sub = self.create_subscription(
            CompressedImage,
            self.radar_image_topic,
            self._on_radar_image_received,
            10,
        )

    def _on_radar_image_received(self, msg: CompressedImage) -> None:
        with self.lock:
            self.latest_radar_image = bytes(msg.data)
            self.latest_radar_image_time = time.monotonic()

    def _start_web_server(self) -> None:
        handler = self._make_request_handler()
        self.http_server = _AgvThreadingHTTPServer(
            (self.web_host, self.web_port),
            handler,
        )
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            name="agv_web_control_http",
            daemon=True,
        )
        self.http_thread.start()

    def _log_web_access_instructions(self) -> None:
        phone_urls = self._get_phone_access_urls()

        if phone_urls:
            self.get_logger().info(
                "En el navegador del celular abre una de estas URLs:"
            )
            for url in phone_urls:
                self.get_logger().info(f"  {url}")
        else:
            self.get_logger().warn(
                "El servidor web esta escuchando solo en esta PC; el celular "
                "no podra abrir el GUI. Usa web_host:=0.0.0.0."
            )

        self.get_logger().info(
            f"En esta PC tambien puedes abrir: http://127.0.0.1:{self.web_port}"
        )

    def _get_phone_access_urls(self) -> list[str]:
        if self.web_host in ("0.0.0.0", "::"):
            urls = {
                f"http://{address}:{self.web_port}"
                for address in self._get_lan_ipv4_addresses()
            }
            urls.add(f"http://{UBUNTU_HOTSPOT_HOST}:{self.web_port}")
            return sorted(urls)

        if self.web_host in ("127.0.0.1", "localhost", "::1"):
            return []

        return [f"http://{self.web_host}:{self.web_port}"]

    @staticmethod
    def _get_lan_ipv4_addresses() -> list[str]:
        addresses: set[str] = set()

        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                address = info[4][0]
                if not address.startswith("127."):
                    addresses.add(address)
        except OSError:
            pass

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                address = sock.getsockname()[0]
                if not address.startswith("127."):
                    addresses.add(address)
        except OSError:
            pass

        return sorted(addresses)

    def _make_request_handler(self) -> type[BaseHTTPRequestHandler]:
        node = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    self._send_html(HTML_PAGE)
                    return

                if path == "/api/state":
                    self._send_json(node.get_state())
                    return

                if path == "/api/radar.jpg":
                    self._send_radar_image()
                    return

                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path != "/api/control":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                try:
                    payload = self._read_json_payload()
                    state = node.handle_browser_command(payload)
                except ValueError as error:
                    self._send_json(
                        {"error": str(error)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return

                self._send_json(state)

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _read_json_payload(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}

                raw_body = self.rfile.read(min(length, 8192))
                try:
                    payload = json.loads(raw_body.decode("utf-8"))
                except json.JSONDecodeError as error:
                    raise ValueError("invalid JSON payload") from error

                if not isinstance(payload, dict):
                    raise ValueError("JSON payload must be an object")

                return payload

            def _send_html(self, body: str) -> None:
                encoded_body = body.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded_body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded_body)

            def _send_json(
                self,
                data: dict[str, Any],
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                encoded_body = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded_body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded_body)

            def _send_radar_image(self) -> None:
                radar_image = node.get_latest_radar_image()
                if radar_image is None:
                    self.send_error(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "No radar image received yet",
                    )
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(radar_image)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(radar_image)

        return RequestHandler

    def get_latest_radar_image(self) -> bytes | None:
        with self.lock:
            return self.latest_radar_image

    def handle_browser_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action", "update")).strip().lower()

        with self.lock:
            self.last_browser_command_time = time.monotonic()

            if action == "stop":
                self._stop_agv_locked()
                self.safety_stop_active = False
                return self._get_state_locked()

            if action == "toggle_mode":
                self._update_controls_from_payload_locked(payload)
                self._toggle_control_mode_locked()
                self.safety_stop_active = False
                return self._get_state_locked()

            if action == "cobot":
                self._toggle_searching_mode_locked()
                self.safety_stop_active = False
                return self._get_state_locked()

            if action == "update":
                self._update_controls_from_payload_locked(payload)
                self.safety_stop_active = False
                return self._get_state_locked()

            if action == "heartbeat":
                return self._get_state_locked()

            raise ValueError(f"unsupported action '{action}'")

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return self._get_state_locked()

    def _update_controls_from_payload_locked(
        self,
        payload: dict[str, Any],
    ) -> None:
        selected_tab = str(
            payload.get("selected_tab", self.selected_tab)
        ).strip()
        if selected_tab in (SELECTED_TAB_SPEED_TURN, SELECTED_TAB_INDEPENDENT):
            self.selected_tab = selected_tab

        self.drive_direction = DRIVE_DIRECTION_FORWARD

        self.speed_percent = self._clamp_int(
            payload.get("speed_percent", self.speed_percent),
            0,
            MAX_SPEED_PERCENT,
        )
        self.turn = self._clamp_int(payload.get("turn", self.turn), -100, 100)
        self.right_wheel = self._clamp_int(
            payload.get("right_wheel", self.right_wheel),
            0,
            MAX_WHEEL_SPEED,
        )
        self.left_wheel = self._clamp_int(
            payload.get("left_wheel", self.left_wheel),
            0,
            MAX_WHEEL_SPEED,
        )
        self.free_stop_active = False

    @staticmethod
    def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
        try:
            numeric_value = int(round(float(value)))
        except (TypeError, ValueError):
            numeric_value = minimum

        return max(minimum, min(numeric_value, maximum))

    def _normalize_wheel_value(self, value: float) -> int:
        wheel_speed = self._clamp_wheel_value(value)

        if wheel_speed == 0:
            return 0

        if wheel_speed < MIN_FORWARD_SPEED:
            return 0

        return wheel_speed

    def _clamp_wheel_value(self, value: float) -> int:
        return int(round(max(0.0, min(float(value), MAX_WHEEL_SPEED))))

    def _get_speed_from_percent_locked(self) -> int:
        speed_percent = int(
            round(max(0.0, min(float(self.speed_percent), MAX_SPEED_PERCENT)))
        )

        if speed_percent == 0:
            return 0

        if speed_percent == 1:
            return MIN_FORWARD_SPEED

        speed_range = MAX_WHEEL_SPEED - MIN_FORWARD_SPEED
        return int(
            round(MIN_FORWARD_SPEED + (speed_percent * speed_range / MAX_SPEED_PERCENT))
        )

    def _get_speed_turn_commands_locked(self) -> tuple[int, int]:
        speed = self._get_speed_from_percent_locked()

        if speed == 0:
            return 0, 0

        turn = float(self.turn)
        if self.invert_turn_direction:
            turn *= -1.0

        turn_ratio = min(abs(turn) / 100.0, 1.0)
        turning_wheel_speed = self._clamp_wheel_value(
            speed * (1.0 - turn_ratio)
        )

        if turn > 0:
            right = turning_wheel_speed
            left = speed
        elif turn < 0:
            right = speed
            left = turning_wheel_speed
        else:
            right = speed
            left = speed

        return right, left

    def _get_independent_wheel_commands_locked(self) -> tuple[int, int]:
        right = self._normalize_wheel_value(self.right_wheel)
        left = self._normalize_wheel_value(self.left_wheel)
        return right, left

    def _get_wheel_commands_locked(self) -> tuple[int, int]:
        if self.selected_tab == SELECTED_TAB_INDEPENDENT:
            return self._get_independent_wheel_commands_locked()

        return self._get_speed_turn_commands_locked()

    def _get_command_label_locked(self) -> str:
        if self.control_mode == CONTROL_MODE_AUTOMATIC:
            return (
                "Command: "
                f"{self.automatic_right_rpm},{self.automatic_left_rpm}   "
                f"right={self.automatic_right_rpm}, "
                f"left={self.automatic_left_rpm}"
            )

        if self.free_stop_active:
            return f"Command: {FREE_MODE_COMMAND}   free mode"

        right, left = self._get_wheel_commands_locked()
        return f"Command: {right},{left}   right={right}, left={left}"

    def _get_state_locked(self) -> dict[str, Any]:
        mode_label = "Mode: MANUAL"
        if self.control_mode == CONTROL_MODE_AUTOMATIC:
            mode_label = "Mode: AUTOMATIC"

        return {
            "control_mode": self.control_mode,
            "mode_label": mode_label,
            "selected_tab": self.selected_tab,
            "drive_direction": self.drive_direction,
            "speed_percent": self.speed_percent,
            "turn": self.turn,
            "right_wheel": self.right_wheel,
            "left_wheel": self.left_wheel,
            "orientation_control_active": self.orientation_control_active,
            "position_control_active": self.position_control_active,
            "harvesting_active": self.harvesting_active,
            "harvesting_refining_active": self.harvesting_refining_active,
            "automatic_right_rpm": self.automatic_right_rpm,
            "automatic_left_rpm": self.automatic_left_rpm,
            "command_label": self._get_command_label_locked(),
            "searching_mode_active": self.searching_mode_active,
            "cobot_label": self._get_cobot_label_locked(),
        }

    def _publish_wheel_command_locked(self, right: int, left: int) -> None:
        msg = String()
        msg.data = f"{right},{left}"
        self.pub_command.publish(msg)

    def _publish_free_mode_command_locked(self) -> None:
        msg = String()
        msg.data = FREE_MODE_COMMAND
        self.pub_command.publish(msg)

    def _publish_control_mode_locked(self) -> None:
        msg = String()
        msg.data = self.control_mode
        self.pub_control_mode.publish(msg)

    def _publish_searching_mode_command_locked(self, command: str) -> None:
        msg = String()
        msg.data = command
        self.pub_searching_mode_cmd.publish(msg)
        self.get_logger().info(
            f"Cobot button -> {self.searching_mode_cmd_topic} {command}"
        )

    def _toggle_searching_mode_locked(self) -> None:
        command = "STOP" if self.searching_mode_active else "START"
        self._publish_searching_mode_command_locked(command)
        self.searching_mode_active = command == "START"
        self.harvesting_active = False
        self.harvesting_refining_active = False

    def _get_cobot_label_locked(self) -> str:
        if self.searching_mode_active:
            return "Stop Cobot"

        return "Cobot"

    def _on_publish_timer(self) -> None:
        if self.is_shutting_down:
            return

        with self.lock:
            if self._browser_command_timed_out_locked():
                self._stop_for_browser_timeout_locked()
                return

            self._publish_control_mode_locked()
            if self.control_mode == CONTROL_MODE_MANUAL:
                if self.free_stop_active:
                    self._publish_free_mode_command_locked()
                else:
                    right, left = self._get_wheel_commands_locked()
                    self._publish_wheel_command_locked(right, left)

    def _browser_command_timed_out_locked(self) -> bool:
        if self.last_browser_command_time is None:
            return False

        elapsed_sec = time.monotonic() - self.last_browser_command_time
        return elapsed_sec > self.command_timeout_sec

    def _stop_for_browser_timeout_locked(self) -> None:
        if not self.safety_stop_active:
            self.get_logger().warn(
                "Browser command timeout; switching to manual and stopping AGV"
            )

        self._stop_agv_locked()
        self.safety_stop_active = True

    def _stop_agv_locked(self) -> None:
        self._set_control_mode_locked(CONTROL_MODE_MANUAL)
        self.selected_tab = SELECTED_TAB_SPEED_TURN
        self.drive_direction = DRIVE_DIRECTION_FORWARD
        self.speed_percent = 0
        self.turn = 0
        self.right_wheel = 0
        self.left_wheel = 0
        self.free_stop_active = True
        self._publish_free_mode_command_locked()

    def _toggle_control_mode_locked(self) -> None:
        if self.control_mode == CONTROL_MODE_MANUAL:
            self._set_control_mode_locked(CONTROL_MODE_AUTOMATIC)
        else:
            self._set_control_mode_locked(CONTROL_MODE_MANUAL)

    def _set_control_mode_locked(self, control_mode: str) -> None:
        if control_mode not in (CONTROL_MODE_MANUAL, CONTROL_MODE_AUTOMATIC):
            return

        self.control_mode = control_mode
        self.free_stop_active = False
        self._publish_control_mode_locked()
        if control_mode == CONTROL_MODE_AUTOMATIC:
            self._publish_wheel_command_locked(0, 0)
        else:
            self._clear_automatic_status_locked()

    def _on_control_status_received(self, msg: String) -> None:
        try:
            values = self._parse_key_value_message(msg.data)
            orientation_control_active = self._parse_bool(
                values.get("orientation_active")
            )
            position_control_active = self._parse_bool(
                values.get("position_active")
            )
            automatic_right_rpm = int(values.get("right_rpm", "0"))
            automatic_left_rpm = int(values.get("left_rpm", "0"))

        except ValueError as error:
            self.get_logger().warn(
                f"Ignoring invalid control status message: {error}"
            )
            return

        with self.lock:
            self.orientation_control_active = orientation_control_active
            self.position_control_active = position_control_active
            self.automatic_right_rpm = automatic_right_rpm
            self.automatic_left_rpm = automatic_left_rpm

    def _on_searching_mode_status_received(self, msg: String) -> None:
        status = msg.data.strip().upper()
        if status == "BUSY":
            searching_mode_active = True
            harvesting_active = False
            harvesting_refining_active = False
        elif status == "REFINING":
            searching_mode_active = True
            harvesting_active = False
            harvesting_refining_active = True
        elif status == "HARVESTING":
            searching_mode_active = True
            harvesting_active = True
            harvesting_refining_active = False
        elif status in ("IDLE", "DONE_OK", "DONE_FAIL"):
            searching_mode_active = False
            harvesting_active = False
            harvesting_refining_active = False
        else:
            return

        with self.lock:
            self.searching_mode_active = searching_mode_active
            self.harvesting_active = harvesting_active
            self.harvesting_refining_active = harvesting_refining_active

    def _clear_automatic_status_locked(self) -> None:
        self.position_control_active = False
        self.orientation_control_active = False
        self.automatic_right_rpm = 0
        self.automatic_left_rpm = 0

    @staticmethod
    def _parse_key_value_message(data: str) -> dict[str, str]:
        values = {}

        for item in data.split(","):
            if not item:
                continue

            key, separator, value = item.partition("=")
            if not separator:
                raise ValueError(f"missing '=' in '{item}'")

            values[key.strip()] = value.strip()

        return values

    @staticmethod
    def _parse_bool(value: str | None) -> bool:
        if value is None:
            return False

        normalized_value = value.strip().lower()
        if normalized_value in ("true", "1", "yes", "on"):
            return True

        if normalized_value in ("false", "0", "no", "off"):
            return False

        raise ValueError(f"invalid boolean value '{value}'")

    def shutdown(self) -> None:
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        with self.lock:
            self._stop_agv_locked()

        if hasattr(self, "http_server"):
            self.http_server.shutdown()
            self.http_server.server_close()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)

    node = None

    try:
        node = AgvWebControlNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
