import wx

        
class ADB_Frame(wx.Frame):
    def __init__(self):
        print "Create ADB GUI"
        wx.Frame.__init__(self, parent = None, title = "ADB monitor", size = (600,600))
        self.counter = 0

        #Load adb command
        
        
        
        
        #Create panel
        self.panel = wx.Panel(self)
        self.bind_component()

        #show value
        
        self.label = wx.StaticText(self.panel, label = "Hello World", pos = (100,50))
        
    def bind_component(self):
        #Button
        self.connect_btn = wx.Button(parent = self.panel, label = "Connect", pos = (300, 100))

        #Bind function
        self.Bind(wx.EVT_BUTTON, self.Btn_connect_func, self.connect_btn)
        
    def Btn_connect_func(self, event):
        print "Click"
        self.counter += 1
        self.label.SetLabel(str(self.counter))
                
def main():
    app = wx.App(False)
    
    frame = ADB_Frame()
    frame.Show()

    app.MainLoop()

if __name__ == "__main__":
    main()


