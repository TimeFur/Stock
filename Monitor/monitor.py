import wx

class ADB_App(wx.Frame):
    def __init__(self):
        print "Create ADB GUI"
        wx.Frame.__init__(self, parent = None, title = "ADB monitor", size = (600,600))

        #Create panel
        self.panel = wx.Panel(self)
        self.bind_component()
        
    def bind_component(self):
        #Button
        self.connect_btn = wx.Button(parent = self.panel, label = "Connect", pos = (300, 100))

        #Bind function
        self.Bind(wx.EVT_BUTTON, self.Btn_connect_func, self.connect_btn)
        
    def Btn_connect_func(self, event):
        print "Click"
        
def main():
    app = wx.PySimpleApp()
    
    frame = ADB_App()
    frame.Show()

    app.MainLoop()
    

if __name__ == "__main__":
    main()


