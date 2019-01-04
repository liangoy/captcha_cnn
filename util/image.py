from captcha.image import ImageCaptcha
import numpy as np
import time

image = ImageCaptcha()
w = [chr(i) for i in range(65, 91)]+[chr(i) for i in range(97, 123)]
d = [chr(i) for i in range(48, 58)]

all = w + d

w2d={}
for i,j in enumerate(all[26:]):
    w2d[j]=i

for i,j in enumerate(all[:26]):
    w2d[j]=i

w2d_func=np.frompyfunc(lambda x:w2d[x],1,1)

def generate_image(num=None):
    cap = np.random.choice(all, size=[num, 4])
    ima = np.array([np.array(image.generate_image(''.join(i)).convert('L')) for i in cap])
    return w2d_func(cap), ima

def generate_number_image(num=None):
    number_list=[]
    captcha_list=[]
    for i in range(num):
        number=np.random.randint(0,10,4)
        captcha=np.array(image.generate_image(''.join([str(i)for i in number])).convert('L'))
        number_list.append(number)
        captcha_list.append(np.array(captcha))
    return number_list,captcha_list

if __name__ == '__main__':
    s = time.time()
    print(np.array(generate_image(2)[1][1].shape))
    e = time.time()
    print(e - s)
