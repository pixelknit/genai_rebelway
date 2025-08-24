bl_info = {
    "name": "LLM Command Box",
    "blender": (3,0,0),
    "category": "Object",
}

import bpy
import os
import requests
import json
import re

def llm_local_to_bpy(command_text: str) -> str:
    prompt = f"""
Translate this natural languange into Blender Python (bpy) code.
Only output code.
-Do not include functions or return statements
-Do not include explanations, only raw bpy code
User: {command_text}
Output:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt}
    )

    output = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                output += data["response"]

    return output.strip()

def clean_code(code: str) -> str:
    code = re.sub(f"^```[a-z]*","",code.strip(), flags=re.MULTILINE)
    code = re.sub(r"```$", "", code.strip(), flags=re.MULTILINE)
    return code.strip()

class OBJECT_OT_llm_command(bpy.types.Operator):
    bl_idname = "object.llm_command"
    bl_label = "Execute LLM Command"

    command: bpy.props.StringProperty(name="Command")

    def execute(self, context):
        cmd = self.command.strip()
        if not cmd:
            self.report({'WARNING'}, "No command entered")
            return {'CANCELED'}
        
        try:
            code = llm_local_to_bpy(cmd)
            print("RAW LLM Output:\n", code)

            safe_code = clean_code(code)
            print("Executing Clened Code:\n", safe_code)

            exec(safe_code, {"bpy":bpy})
        except Exception as e:
            self.report({'ERROR'}, f"Execution failed: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}
    
class OBJECT_PT_llm_command_panel(bpy.types.Panel):
    bl_label = "LLM Command Box"
    bl_idname = "OBJECT_PT_llm_command_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Commands"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Enter a llm command:")
        layout.prop(context.scene, "llm_command_input")
        layout.operator("object.llm_command", text="Run").command = context.scene.llm_command_input

def register():
    bpy.utils.register_class(OBJECT_OT_llm_command)
    bpy.utils.register_class(OBJECT_PT_llm_command_panel)
    bpy.types.Scene.llm_command_input = bpy.props.StringProperty(
        name="Command Input",
        default=""
    )

def uregister():
    bpy.utils.unregister_class(OBJECT_OT_llm_command)
    bpy.utils.unregister_class(OBJECT_PT_llm_command_panel)
    del bpy.types.Scene.llm_command_input

if __name__ == "__main__":
    register() 
