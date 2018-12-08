# -*- coding:utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

import tkinter
from PIL import Image, ImageDraw,ImageTk

class MyChain(chainer.Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            conv1=L.Convolution2D(1, 16, 5, 1, 0), # 1層目の畳み込み層（チャンネル数は16）
            conv2=L.Convolution2D(16, 64, 5, 1, 0), # 2層目の畳み込み層（チャンネル数は32）
            l3=L.Linear(None, 10), #クラス分類用
        )

    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        y = self.l3(h2)
        return y    


class Scribble(object):

    #クリックされた
    def on_clicked(self, event):
        self.sx = event.x
        self.sy = event.y

    # ドラッグされた
    def on_dragged(self, event):
        self.canvas.create_line(self.sx, self.sy, event.x, event.y,
                                fill = self.color,
                                width = self.width)
        self.draw.line([self.sx, self.sy,event.x,  event.y], fill = self.color,width = self.width)
        self.sx = event.x
        self.sy = event.y

    #分類ボタン
    def classification_button(self,event):
        #画像書き出し
        filename = "my_drawing.png"
        self.image1.save(filename)
        print("quit")
        img = Image.open("my_drawing.png")
        
        img = img.convert('L') # グレースケール変換
        img = img.resize((28, 28)) # 28x28にリサイズ
        
        imgshow = img.resize((300, 300)) # 8x8にリサイズ
        
        imgshow.save("input.png","png",quality=100, optimize=True)
        img = 16.0 - np.asarray(img, dtype=np.float32) / 16.0 # 白黒反転，0〜15に正規化，array化
        img = img[np.newaxis, np.newaxis, :, :] # 4次元行列に変換（1x1x8x8，バッチ数xチャンネル数x縦x横）
        self.classification(img)

        input_image = ImageTk.PhotoImage(file="input.png",width=300,height=300)
        try:
            self.canvas.create_image((300,0),image=input_image)
        except Exception as e:
            print(e.args)

    #クリアボタン
    def clear_button(self,event):
        self.clear()     
       
    #終了ボタン
    def quit_button(self,event):
       self.window.quit()
       quit

    # ウィンドウを作る 
    def create_window(self):
        window = tkinter.Tk()
        self.window_width = 600
        self.window_height = 600
        self.canvas = tkinter.Canvas(window, bg = "white",width=self.window_width,height=self.window_height)
        self.canvas.pack()
        self.image1 = Image.new("RGB", (int(self.window_width/2), int(self.window_height/2)), "white")
        self.draw = ImageDraw.Draw(self.image1)
        
        Static = []
        for i in range(10):
            Static.append( tkinter.Label(text=str(i), foreground="black",font=("",20)))
            Static[i].place(x=self.window_width/11*(i+1)-10,  y=self.window_height-30, )

        classify_button = tkinter.Button(window, text = "Classification")
        classify_button.bind("<Button-1>",self.classification_button)
        classify_button.pack(side = tkinter.RIGHT)
        clear_button = tkinter.Button(window, text = "Clear")
        clear_button.bind("<Button-1>",self.clear_button)
        clear_button.pack(side = tkinter.RIGHT)
        quiter = tkinter.Button(window, text = "Quit")
        quiter.bind("<Button-1>",self.quit_button)
        quiter.pack(side = tkinter.LEFT)

        self.canvas.bind("<B1-Motion>", self.on_dragged)
        self.canvas.bind("<ButtonPress-1>", self.on_clicked)


        # 色の設定                    
        self.color="black"                             

        # 線の太さ設定
        self.width = 30

        self.setline
        
        return window;

    def classification(self,img):
        x = chainer.Variable(img)
        y = self.model.predictor(x)
        c = F.softmax(y).data.argmax()
        print(c)

        #出力の正規化
        y1=y.data[0]
        ymin=min(y1)
        ymax=max(y1)
        value_range=ymax-ymin
        y1= list(map(lambda x :(x-ymin)/value_range*300,y1))

        #グラフの描画
        #左端
        self.canvas.create_line(0, 600, self.window_width/11*(1), 600-y1[0],
                                fill = self.color,
                                width = 2)
        for i in range(0,9):
            if y1[i]==300:
                maxindex = i
            self.canvas.create_line(self.window_width/11*(i+1), 600-y1[i], self.window_width/11*(i+2), 600-y1[i+1],
                                fill = self.color,
                                width = 2)

        #右端
        self.canvas.create_line(self.window_width/11*(10), 600-y1[9], self.window_width, 600,
                                fill = self.color,
                                width = 2)
        #最大値
        self.canvas.create_line(self.window_width/11*(maxindex+1), 300, self.window_width/11*(maxindex+1),600,
                                fill = "red",
                                width = 3)
    
    #描画の初期化
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,self.window_height/2,self.window_width/2],fill=(255,255,255))
        self.setline()
    def setline(self):
        self.canvas.create_line(0,self.window_height/2,self.window_width,self.window_height/2,fill="black",width = 5)
        self.canvas.create_line(self.window_width/2,0,self.window_width/2,self.window_height/2,fill="black",width = 5)
    
    
    def __init__(self):
        self.window = self.create_window(); 
        self.setline()
        self.model = L.Classifier(MyChain(), lossfun=F.softmax_cross_entropy)
        self,chainer.serializers.load_npz("result/CNN.model", self.model)
        
    def run(self):
        self.window.mainloop()
       

Scribble().run()