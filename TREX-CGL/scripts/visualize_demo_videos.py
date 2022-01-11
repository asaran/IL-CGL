import numpy as np
import cv2
import argparse


parser = argparse.ArgumentParser(description=None)
parser.add_argument('--data_dir', default='../../learning-rewards-of-learners/data/atari-head/', help='data directory')
parser.add_argument('--env_name', default='mspacman')
parser.add_argument('--trial_name', default='209_RZ_6964528_Jan-08-10-23-46')
parser.add_argument('--out_path', default='gaze_videos/experts/')

args = parser.parse_args()
env_name = args.env_name
data_dir = args.data_dir
trial_name = args.trial_name
out_path = args.out_path

#trial = '../data/atari-head/mspacman/209_RZ_6964528_Jan-08-10-23-46'
trial = data_dir + '/' + env_name + '/' + trial_name

img_folder = trial + '/'
#img_folder = '../data/atari-head/hero/195_RZ_205678_Jun-28-12-09-00/'
f = open(trial+".txt")
fps = 20
format="XVID"
outvid= out_path+'/'+trial_name+'.avi'
print(outvid)

fourcc = cv2.VideoWriter_fourcc(*format)
vid = None
size = None
is_color = True


action_name = { 
'0': 'PLAYER_A_NOOP',

'1': 'PLAYER_A_FIRE',          
'2': 'PLAYER_A_UP',             
'3': 'PLAYER_A_RIGHT',          
'4': 'PLAYER_A_LEFT',           
'5': 'PLAYER_A_DOWN',          

'6': 'PLAYER_A_UPRIGHT',        
'7': 'PLAYER_A_UPLEFT',         
'8': 'PLAYER_A_DOWNRIGHT',     
'9': 'PLAYER_A_DOWNLEFT',       

'10': 'PLAYER_A_UPFIRE',        
'11': 'PLAYER_A_RIGHTFIRE',     
'12': 'PLAYER_A_LEFTFIRE',      
'13': 'PLAYER_A_DOWNFIRE',     

'14': 'PLAYER_A_UPRIGHTFIRE',   
'15': 'PLAYER_A_UPLEFTFIRE',    
'16': 'PLAYER_A_DOWNRIGHTFIRE', 
'17': 'PLAYER_A_DOWNLEFTFIRE',
'null': 'NULL'
}

font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText    = (10,10)
bottomLeftCornerOfText = (10,180)
bottomRightCornerOfText = (130,180)
fontScale              = 0.25
fontColor              = (255,255,255)
lineType               = 1

line = f.readline()
i = 0
blink = False
counter = 0
colors = [(0,255,0),(0,0,255)]

k=0
gaze_ignore = []
for line in f:
	k+=1
	gaze_ignore.append(0)
	contents = line.split(',')

	img_name = contents[0]
	episode = contents[1]
	score = contents[2]
	duration = contents[3]
	unclipped_reward = contents[4]
	action = contents[5]
	gaze = contents[6:]

	img = cv2.imread(img_folder+img_name+'.png')

	for j in range(0,len(gaze),2):
		if('null' not in gaze[j]):
			x = float(gaze[j])
			y = float(gaze[j+1])
		if(y>200):
			blink = True
	if blink:	
		blink=False 

f.close()
f = open(trial+".txt")
line = f.readline()
blink = False
k = 0
for line in f:	
	contents = line.split(',')
	img_name = contents[0]
	episode = contents[1]
	score = contents[2]
	duration = contents[3]
	unclipped_reward = contents[4]
	action = contents[5]
	gaze = contents[6:]

	img = cv2.imread(img_folder+img_name+'.png')

	if vid is None:
		if size is None:
			size = img.shape[1], img.shape[0]
		vid = cv2.VideoWriter(outvid, fourcc, float(fps), size, is_color)


	# overlay gaze coordinates on img
	for j in range(0,len(gaze),2):
		if('null' not in gaze[j]):
			x = float(gaze[j])
			y = float(gaze[j+1])
		if(y>200):
			blink = True 
			counter = 0
			text_color = colors[(i+1)%2]
		if blink:
			counter+=1	
		if counter==300 and blink==True:
			blink=False

		gaze_coord_text = '('+str(int(x))+','+str(int(y))+')'
		
		if gaze_ignore[k]==0:
			cv2.circle(img, (int(x),int(y)), 5, (0,255,0), thickness=1, lineType=8, shift=0)

		cv2.putText(img,action_name[action], 
			topLeftCornerOfText, 
			font, 
			fontScale,
			fontColor,
			lineType)
	k+=1

	vid.write(img)
vid.release()