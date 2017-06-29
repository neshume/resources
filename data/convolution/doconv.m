f=fopen('input');         x    = fread(f,inf,'single'); fclose(f);
f=fopen('conv0');         y_gt = fread(f,inf,'single'); fclose(f);
f=fopen('conv0.biases');  b    = fread(f,inf,'single'); fclose(f);
f=fopen('conv0.weights'); w    = fread(f,inf,'single'); fclose(f);

%outputXhXwXinput

x     = reshape(x,   [16 256 256 1 ]);
y_gt  = reshape(y_gt,[16 256 256 1 ]);
b     = reshape(b,   [16 1   1   1 ]);
w     = reshape(w,   [16 3   3   16]);

x = padarray(x,[0 1 1 0],0,'both');
pad = 1;

for k = 1:16
    for i = 1:256
        for j = 1:256
            xblob = x(:,  pad+i+(-1:1),  pad+j+(-1:1),  1);
            wblob = w(:,:,:,k);
            bblob = b(k);
            y(k,i,j) = dot(xblob(:),wblob(:))+bblob;
        end
    end
    fprintf('Done: %i\n',k);
end

