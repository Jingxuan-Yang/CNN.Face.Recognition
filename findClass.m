function  className = findClass(c_name)

% find class name

class = {'KA','KL','KM','KR','MK','NA','NM','TM','UY','YM','JX','YJX',...
         'XYX','ZMZ','ZQL','XF','WH'};

i = 1;

while class(i) ~= c_name
    
    i = i + 1;
    
end

className = class(i);
