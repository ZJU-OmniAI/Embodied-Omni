from ai2thor.controller import Controller


class BaseAction:
    def __init__(self):
        self.action_mapping = {
            # move agent
            "stand": self.stand,
            "crouch": self.crouch,
            "move_ahead": self.move_ahead,
            "move_back": self.move_back,
            "move_left": self.move_left,
            "move_right": self.move_right,
            "teleport": self.teleport,
            "rotate_left": self.rotate_left,
            "rotate_right": self.rotate_right,
            "look_up": self.look_up,
            "look_down": self.look_down,
            # move arm
            # "move_arm": self.move_arm,
            # "set_hand_radius": self.set_hand_radius,
            # "arm_reset": self.arm_reset,
            # move object
            "pick_up": self.pick_up,
            "release": self.release,
            "put_in": self.put_in,
            "drop_out": self.drop_out,
            "throw_out": self.throw_out,
            "move_hand_object": self.move_hand_object,
            "rotate_hand_object": self.rotate_hand_object,
            # change object state
            "open": self.open,
            "close": self.close,
            "break_": self.break_,
            "cook": self.cook,
            "slice_": self.slice_,
            "toggle_on": self.toggle_on,
            "toggle_off": self.toggle_off,
            "dirty": self.dirty,
            "clean": self.clean,
            "fill": self.fill,
            "empty": self.empty,
            "use_up": self.use_up,
        }

    # --Move agent------------------------------------------------------------------------------------------------------------#
    @staticmethod
    def stand(controller):
        return controller.step(
            action="Stand",
        )

    @staticmethod
    def crouch(controller):
        return controller.step(
            action="Crouch",
        )

    @staticmethod
    def move_ahead(controller: Controller, moveMagnitude=0.25):
        return controller.step(
            action="MoveAhead",
            moveMagnitude=moveMagnitude,
            # returnToStart=True,
        )

    @staticmethod
    def move_back(controller, moveMagnitude=0.25):
        return controller.step(
            action="MoveBack",
            moveMagnitude=moveMagnitude,
            # returnToStart=True,
        )

    @staticmethod
    def move_left(controller, moveMagnitude=0.25):
        controller.step(action="MoveLeft", moveMagnitude=moveMagnitude)

    @staticmethod
    def move_right(controller, moveMagnitude=0.25):
        controller.step(action="MoveRight", moveMagnitude=moveMagnitude)

    @staticmethod
    def teleport(controller, position, rotation):
        return controller.step(
            action="Teleport",
            position=position,
            rotation=rotation,
            horizon=60,
        )

    @staticmethod
    def rotate_left(controller, degrees=90):

        return controller.step(action="RotateLeft", degrees=degrees)

    @staticmethod
    def rotate_right(controller, degrees=90):

        return controller.step(action="RotateRight", degrees=degrees)

    @staticmethod
    def look_up(controller, degrees=10):
        return controller.step(action="LookUp", degrees=degrees)

    @staticmethod
    def look_down(controller, degrees=10):
        return controller.step(action="LookDown", degrees=degrees)

    # --Move agent------------------------------------------------------------------------------------------------------------#

    # --Move arm--------------------------------------------------------------------------------------------------------------#
    # @staticmethod
    # def move_arm( controller, position):
    #     return controller.step(
    #         action="MoveArm",
    #         position=dict(x=0, y=0.5, z=0),
    #         coordinateSpace="armBase",
    #         restrictMovement=False,
    #         speed=1,
    #         returnToStart=True,
    #         fixedDeltaTime=0.02
    #     )

    # @staticmethod
    # def set_hand_radius( controller, radius=0.1):
    #     '''
    #     The radius on the agent's "magnet sphere" hand, in meters.
    #     Valid values are in (0.04:0.5)meters.
    #     '''
    #     return controller.step(
    #         action="SetHandSphereRadius",
    #         radius=radius
    #     )

    # @staticmethod
    # def arm_reset(controller):
    #     try:
    #         return controller.step(
    #             action="MoveArm",
    #             position=dict(x=0, y=0, z=-1),
    #             coordinateSpace="armBase",
    #             restrictMovement=False,
    #             speed=1,
    #             returnToStart=True,
    #             fixedDeltaTime=0.02
    #         )
    #     except Exception as e:
    #         print(e)
    #         return None
    # --Move arm--------------------------------------------------------------------------------------------------------------#

    # --Move object-----------------------------------------------------------------------------------------------------------#
    @staticmethod
    def pick_up(controller, object_id):
        # Other supported directions
        # controller.step("MoveHeldObjectBack")
        # controller.step("MoveHeldObjectLeft")
        # controller.step("MoveHeldObjectRight")
        # controller.step("MoveHeldObjectUp")
        # controller.step("MoveHeldObjectDown")

        return controller.step(
            action="PickupObject",
            objectId=object_id,
            forceAction=True,
            manualInteract=False,
        )

    @staticmethod
    def release(
        controller,
    ):
        return controller.step(action="ReleaseObject")

    @staticmethod
    def put_in(controller, object_id):
        """
        The PutObject command places an object picked up by the PickupObject action onto a receptacle typed object specified by objectId.
        """
        return controller.step(
            action="PutObject",
            objectId=object_id,
            forceAction=True,
            placeStationary=True,
        )

    @staticmethod
    def drop_out(controller):
        return controller.step(action="DropHandObject", forceAction=False)

    @staticmethod
    def throw_out(controller):
        return controller.step(
            action="ThrowObject", moveMagnitude=150.0, forceAction=False
        )

    def move_hand_object(controller, ahead=0.1, right=0.05, up=0.12):
        # controller.step(
        #     action="MoveHeldObjectAhead",
        #     moveMagnitude=moveMagnitude,
        #     forceVisible=False
        # )
        # # Other supported directions
        # controller.step("MoveHeldObjectBack")
        # controller.step("MoveHeldObjectLeft")
        # controller.step("MoveHeldObjectRight")
        # controller.step("MoveHeldObjectUp")
        # controller.step("MoveHeldObjectDown")
        return controller.step(
            action="MoveHeldObject", ahead=ahead, right=right, up=up, forceVisible=False
        )

    @staticmethod
    def rotate_hand_object(controller, pitch=90, yaw=25, roll=45):
        return controller.step(
            action="RotateHeldObject",
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            # rotation=dict(x=90, y=15, z=25)
        )

    # --Move object------------------------------------------------------------------------------------------------------------#

    # --Change object state----------------------------------------------------------------------------------------------------#
    @staticmethod
    def open(controller, object_id):
        isAgentPickup = False
        pickupobjId = ""
        for obj in controller.last_event.metadata["objects"]:
            if obj["isPickedUp"] == True:
                isAgentPickup = True
                pickupobjId = obj["objectId"]

        if isAgentPickup == True:
            for obj in controller.last_event.metadata["objects"]:
                if obj["objectType"] == "Floor":
                    floorid = obj["objectId"]
            controller.step(
                action="PutObject",
                objectId=floorid,
                forceAction=True,
                placeStationary=True,
            )
            controller.step(
                action="OpenObject",
                objectId=object_id,  # "Book|0.25|-0.27|0.95",
                openness=1,
                forceAction=False,
            )
            return controller.step(
                action="PickupObject",
                objectId=pickupobjId,
                forceAction=True,
                manualInteract=False,
            )

        else:
            return controller.step(
                action="OpenObject",
                objectId=object_id,  # "Book|0.25|-0.27|0.95",
                openness=1,
                forceAction=False,
            )

    @staticmethod
    def close(controller, object_id, force_action=False):
        isAgentPickup = False
        pickupobjId = ""
        for obj in controller.last_event.metadata["objects"]:
            if obj["isPickedUp"] == True:
                isAgentPickup = True
                pickupobjId = obj["objectId"]

        if isAgentPickup == True:
            for obj in controller.last_event.metadata["objects"]:
                if obj["objectType"] == "Floor":
                    floorid = obj["objectId"]
            controller.step(
                action="PutObject",
                objectId=floorid,
                forceAction=True,
                placeStationary=True,
            )
            controller.step(
                action="CloseObject",
                objectId=object_id,  # "Book|0.25|-0.27|0.95",
                forceAction=force_action,
            )
            return controller.step(
                action="PickupObject",
                objectId=pickupobjId,
                forceAction=True,
                manualInteract=False,
            )

        return controller.step(
            action="CloseObject",
            objectId=object_id,  # "Book|0.25|-0.27|0.95",
            forceAction=force_action,
        )

    @staticmethod
    def break_(controller, object_id):
        return controller.step(
            action="BreakObject", objectId=object_id, forceAction=False
        )

    @staticmethod
    def cook(controller, object_id):
        return controller.step(
            action="CookObject", objectId=object_id, forceAction=False
        )

    @staticmethod
    def slice_(controller, object_id="Potato|0.25|-0.27|0.95"):
        isAgentPickup = False
        pickupobjId = ""
        for obj in controller.last_event.metadata["objects"]:
            if obj["isPickedUp"] == True:
                isAgentPickup = True
                pickupobjId = obj["objectId"]

        if isAgentPickup == True:
            for obj in controller.last_event.metadata["objects"]:
                if obj["objectType"] == "Floor":
                    floorid = obj["objectId"]
            controller.step(
                action="PutObject",
                objectId=floorid,
                forceAction=True,
                placeStationary=True,
            )
            controller.step(
                action="SliceObject",
                objectId=object_id,  # "Book|0.25|-0.27|0.95",
                forceAction=True,
            )
            return controller.step(
                action="PickupObject",
                objectId=pickupobjId,
                forceAction=True,
                manualInteract=False,
            )

        return controller.step(
            action="SliceObject", objectId=object_id, forceAction=True
        )

    @staticmethod
    def toggle_on(controller, object_id):
        return controller.step(
            action="ToggleObjectOn", objectId=object_id, forceAction=True
        )

    @staticmethod
    def toggle_off(controller, object_id):
        return controller.step(
            action="ToggleObjectOff", objectId=object_id, forceAction=True
        )

    @staticmethod
    def dirty(controller, object_id):
        return controller.step(
            action="DirtyObject", objectId=object_id, forceAction=False
        )

    @staticmethod
    def clean(controller, object_id):
        return controller.step(
            action="CleanObject", objectId=object_id, forceAction=False
        )

    @staticmethod
    def fill(controller, object_id):
        return controller.step(
            action="FillObjectWithLiquid", objectId=object_id, forceAction=False
        )

    @staticmethod
    def empty(controller, object_id):
        return controller.step(
            action="EmptyLiquidFromObject", objectId=object_id, forceAction=False
        )

    @staticmethod
    def use_up(controller, object_id):
        return controller.step(
            action="UseUpObject", objectId=object_id, forceAction=False
        )

    # --Change object state----------------------------------------------------------------------------------------------------------------------#
