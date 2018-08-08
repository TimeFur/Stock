import wx


'''================================
            DEFINE
================================'''
RX = 36
TX = 18

START_ROW = 10
START_COL = 10
BLOCK_SIZE_HEIGHT = 28
BLOCK_SIZE_VERTICE = 26
VAL_SIZE    = 8

class ADB_Frame(wx.Frame):
    def __init__(self):
        print "Create ADB GUI"
        wx.Frame.__init__(self, parent = None, title = "ADB monitor", size = (1000,1000))
        self.counter = 1000

        #Load adb command
                
        #Create panel
        self.panel = wx.Panel(self)
        self.bind_component()

        #show value
        self.label = []
        self.create_screen()
        
    def bind_component(self):
        #Button
        self.connect_btn = wx.Button(parent = self.panel, label = "Connect", pos = (500, 500))

        #Bind function
        self.Bind(wx.EVT_BUTTON, self.Btn_connect_func, self.connect_btn)
        
    def Btn_connect_func(self, event):
        print "Click"
        self.counter += 1
        self.label[0][1].SetLabel(str(self.counter))

    def create_screen(self):
        for i in range(RX):
            tmp = []
            for j in range(TX):
                block = wx.StaticText(self.panel,
                                         label = "0",
                                         pos = (START_ROW + BLOCK_SIZE_HEIGHT * i, START_COL + BLOCK_SIZE_VERTICE * j))
                block_format = wx.Font(VAL_SIZE, wx.DEFAULT, wx.NORMAL, wx.BOLD)
                block.SetFont(block_format)
                
                tmp.append(block)

            self.label.append(tmp)
            
def main():
    app = wx.App(False)
    
    frame = ADB_Frame()
    frame.Show()

    app.MainLoop()

if __name__ == "__main__":
    main()


