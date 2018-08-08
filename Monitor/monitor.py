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
        #ADB connect Button
        self.connect_btn = wx.Button(parent = self.panel, label = "Connect", pos = (0, 500))
        self.Bind(wx.EVT_BUTTON, self.Btn_connect_func, self.connect_btn)

        self.connect_rb1 = wx.RadioButton(self.panel, 11, label = 'USB', pos = (100,505), style = wx.RB_GROUP) 
        self.connect_rb2 = wx.RadioButton(self.panel, 22, label = 'WIFI',pos = (150,505)) 
        
        #ADB connect Button
        self.disconnect_btn = wx.Button(parent = self.panel, label = "Disconnect", pos = (0, 530))
        self.Bind(wx.EVT_BUTTON, self.Btn_disconnect_func, self.disconnect_btn)
        self.connect_rb2 = wx.RadioButton(self.panel, 22, label = 'USB', pos = (100,535), style = wx.RB_GROUP)
        
    def Btn_connect_func(self, event):
        print "Click"
        print self.connect_rb1.GetValue()
        print self.connect_rb2.GetValue()
        
        self.counter += 1
        self.label[0][1].SetLabel(str(self.counter))
        
    def Btn_disconnect_func(self, event):
        print "DisClick"
        self.counter += 1
        self.label[0][2].SetLabel(str(self.counter))

    #Whole block
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


