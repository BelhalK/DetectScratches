-- 
require 'image'
require 'nn'

-- Rescales and normalizes the image
function preprocess(im, img_mean)
  -- rescale the image
  local im3 = image.scale(im,224,224,'bilinear')
  -- subtract imagenet mean and divide by std
  for i=1,3 do im3[i]:add(-img_mean.mean[i]):div(img_mean.std[i]) end
  return im3
end


print '==> Downloading image and network'
local image_url = arg[1]
-- local image_url = '/neck.jpg'
local network_url = '/nin_nobn_final.t7'
local image_name = paths.basename(image_url)
local network_name = paths.basename(network_url)
if not paths.filep(image_name) then os.execute('wget '..image_url)   end
if not paths.filep(network_name) then os.execute('wget '..network_url)   end

print '==> Loading network'
local net = torch.load(network_name):unpack():float()

-- print the neural net and its training set in the .t7
-- print(torch.load(network_name))
-- assert(false)


net:evaluate()
print(net)

print '==> Loading synsets'
print 'Loads mapping from net outputs to human readable labels'
local synset_words = {}
for line in io.lines'synset_words.txt' do table.insert(synset_words, line:sub(11)) end

print '==> Loading image and imagenet mean'
local im = image.load(image_name)

print '==> Preprocessing'
-- Our network has mean and std saved in net.transform
print('ImageNet ', net.transform)
local I = preprocess(im, net.transform):view(1,3,224,224):float()

print 'Propagate through the network, sort outputs in decreasing order and show 5 best classes'
local _,classes = net:forward(I):view(-1):sort(true)
for i=1,5 do
  print('predicted class '..tostring(i)..': ', synset_words[classes[i] ])
end


print('the class of this image is', synset_words[classes[1]])


