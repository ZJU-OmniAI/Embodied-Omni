import math
from .base_agent import BaseAgent

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class RocAgent(BaseAgent):
    def __init__(self, controller):
        super().__init__(controller)
        self.controller = controller
        self.legal_location = {}
        self.agent_state = []
        self.object_state = {}

    def observe_once(self, leftright="left", degrees="80"):
        isAgentPickup = False
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["isPickedUp"]:
                isAgentPickup = True

        if leftright == "left":
            self.action.action_mapping["rotate_left"](self.controller, degrees=degrees)
            if isAgentPickup:
                self.controller.step(
                    action="MoveHeldObjectDown", moveMagnitude=0.07, forceVisible=False
                )
                self.controller.step(
                    action="MoveHeldObjectBack", moveMagnitude=0.05, forceVisible=False
                )
        elif leftright == "right":
            self.action.action_mapping["rotate_right"](self.controller, degrees=degrees)
            if isAgentPickup:
                self.controller.step(
                    action="MoveHeldObjectDown", moveMagnitude=0.07, forceVisible=False
                )
                self.controller.step(
                    action="MoveHeldObjectBack", moveMagnitude=0.05, forceVisible=False
                )

    def move_forward(self, distance=1):
        self.action.action_mapping["move_ahead"](self.controller, distance)
        if self.controller.last_event.metadata["errorMessage"] == "":
            return
        self.action.action_mapping["move_right"](self.controller, distance)
        if self.controller.last_event.metadata["errorMessage"] == "":
            return
        self.action.action_mapping["move_left"](self.controller, distance)
        if self.controller.last_event.metadata["errorMessage"] == "":
            return
        self.action.action_mapping["move_back"](self.controller, distance)
        if self.controller.last_event.metadata["errorMessage"] == "":
            return
        self.action.action_mapping["rotate_right"](self.controller, degrees=90)
        self.action.action_mapping["move_ahead"](self.controller, distance)
        if self.controller.last_event.metadata["errorMessage"] == "":
            return
        self.action.action_mapping["rotate_left"](self.controller, degrees=180)
        self.action.action_mapping["move_ahead"](self.controller, distance)
        if self.controller.last_event.metadata["errorMessage"] == "":
            return
        self.action.action_mapping["rotate_left"](self.controller, degrees=90)
        self.action.action_mapping["move_ahead"](self.controller, distance)

    def navigate(self, item):
        target_position, target_rotation = self.compute_position_8(
            item, pre_target_positions=[]
        )
        if target_position is None:
            return False, None, None

        event = self.action.action_mapping["teleport"](
            self.controller, position=target_position, rotation=target_rotation
        )
        pre_target_positions = []
        max_retries = 5
        while not event.metadata["lastActionSuccess"] and max_retries > 0:
            print("teleport failed, retrying...")
            pre_target_positions.append(target_position)
            target_position, target_rotation = self.compute_position_8(
                item, pre_target_positions
            )
            if target_position is None:
                break
            event = self.action.action_mapping["teleport"](
                self.controller, position=target_position, rotation=target_rotation
            )
            self.update_event()
            max_retries -= 1

        return event.metadata["lastActionSuccess"], target_position, target_rotation

    def interact(self, item, interact_type):
        object_id = item["objectId"]
        isAgentPickup = False
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["isPickedUp"]:
                isAgentPickup = True

        if isAgentPickup:
            self.controller.step(
                action="MoveHeldObjectDown", moveMagnitude=0.07, forceVisible=False
            )
            self.controller.step(
                action="MoveHeldObjectBack", moveMagnitude=0.05, forceVisible=False
            )

        if interact_type == "open":
            self.action.action_mapping["open"](self.controller, object_id)
        elif interact_type == "close":
            self.action.action_mapping["close"](self.controller, object_id)
        elif interact_type == "break_":
            self.action.action_mapping["break_"](self.controller, object_id)
        elif interact_type == "cook":
            self.action.action_mapping["cook"](self.controller, object_id)
        elif interact_type == "slice_":
            self.action.action_mapping["slice_"](self.controller, object_id)
        elif interact_type == "toggle_on":
            self.action.action_mapping["toggle_on"](self.controller, object_id)
        elif interact_type == "toggle_off":
            self.action.action_mapping["toggle_off"](self.controller, object_id)
        elif interact_type == "dirty":
            self.action.action_mapping["dirty"](self.controller, object_id)
        elif interact_type == "clean":
            self.action.action_mapping["clean"](self.controller, object_id)
        elif interact_type == "fill":
            self.action.action_mapping["fill"](self.controller, object_id)
        elif interact_type == "empty":
            self.action.action_mapping["empty"](self.controller, object_id)
        elif interact_type == "use_up":
            self.action.action_mapping["use_up"](self.controller, object_id)
        elif interact_type == "pick_up":
            self.action.action_mapping["pick_up"](self.controller, object_id)
        elif interact_type == "put":
            self.action.action_mapping["put_in"](self.controller, object_id)
        else:
            raise ValueError(f"Interact type {interact_type} is not defined.")
