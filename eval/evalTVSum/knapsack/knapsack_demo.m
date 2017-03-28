%%% Knapsack demonstration
% Ïnteger weights of items
N = 12;
weights = randint(1,N,[1 1000])
%%
% Values of the items (don't have to be integers)
values = randint(1,N,[1 100])

%% Solve the knapsack problem
% Call the provided m-file
capacity = 3000;
[best amount] = knapsack(weights, values, capacity);
best
items = find(amount)
%%
% Check that the result matches the contraint and the best value
sum(weights(items))
sum(values(items))