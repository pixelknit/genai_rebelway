bl_info = {
    "name": "Text Command Box",
    "blender": (3,0,0),
    "category": "Object",
}

import bpy

class OBJECT_OT_text_command(bpy.types.Operator):
    bl_idname = "object.text_command"
    bl_label = "Execute Text Command"

    command: bpy.props.StringProperty(name="Command")

    def execute(self, context):
        cmd = self.command.lower().strip()

        if "cube" in cmd:
            bpy.ops.mesh.primitive_cube_add()
        elif "sphere" in cmd:
            bpy.ops.mesh.primitive_uv_sphere_add()
        elif "plane" in cmd:
            bpy.ops.mesh.primitive_plane_add()
        elif "cone" in cmd:
            bpy.ops.mesh.primitive_cone_add()

        else:
            self.report({"WARNING"}, f"Unknown command: {cmd}")

        return {"FINISHED"}
    
class OBJECT_PT_text_command_panel(bpy.types.Panel):
    bl_label = "Text Command Box"
    bl_idname = "OBJECT_PT_text_command_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Commands"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Enter a command:")
        layout.prop(context.scene, "text_command_input")
        layout.operator("object.text_command", text="Run").command = context.scene.text_command_input

def register():
    bpy.utils.register_class(OBJECT_OT_text_command)
    bpy.utils.register_class(OBJECT_PT_text_command_panel)
    bpy.types.Scene.text_command_input = bpy.props.StringProperty(
        name="Command Input",
        default=""
    )

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_text_command)
    bpy.utils.unregister_class(OBJECT_PT_text_command_panel)
    del bpy.types.Scene.text_command_input

if __name__ == "__main__":
    register()