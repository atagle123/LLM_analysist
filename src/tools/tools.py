

class Tools:
    def __init__(self) -> None:
        self.tools=[]
        pass
    def add_tools(self,tool):
        self.tools.append(tool)
        print("Added tool")

    def __call__(self):
        return(self.tools)

