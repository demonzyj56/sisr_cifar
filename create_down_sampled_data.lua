--[[
Create down sampled cifar10 data from files
--]]


require 'image'
counter = 0

function read_image(idx, cache_dir)
    img = image.load(cache_dir..string.format('/%d_down_waifu2x.png', idx-1))
    img = img * 255 
    return img
end

function create_data(num_images, cache_dir)
    local data = torch.Tensor(num_images, 3, 32, 32)
    for i=1,num_images do
        local img = read_image(i, cache_dir)
        -- print(i)
        -- print(img:size(1))
        if img:size(1) == 1 then
            counter = counter + 1
        else
            data[i] = img
        end
    end
    return data
end

num_images = 10000
cache_dir = "cache_test_waifu2x"
data = create_data(num_images, cache_dir)
print(counter)
torch.save("cifar10_test_waifu2x.t7", data)
