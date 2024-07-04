from .interpreter import PythonCodeInterpreter


def tool2json(tool):
    args_schema = tool.args_schema.model_json_schema()
    fmt = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": args_schema['properties'],
                "required": args_schema['required'],
            },
        }
    }
    return fmt


def load_tools(names):
    name2tool = {
        "python_code_interpreter": PythonCodeInterpreter(),
    }
    descriptions = []
    for name in names:
        if name not in name2tool:
            raise NotImplementedError
        tool = name2tool[name]
        descriptions.append(tool2json(tool))
    return name2tool, descriptions