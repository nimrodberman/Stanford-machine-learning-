A = rand(3,4);
disp(A);

disp(max(A,[],2));
disp(max(max(A,[],2)));

[maxval,col] = 