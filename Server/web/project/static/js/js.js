var app = angular.module('instacorrect', []);

app.controller('mainCtrl', function($scope, $http){
  // Normale state
  $scope.state = "idle";
  $scope.corrected_sentence = "";
  $scope.correct = function(){
    $scope.state = "loading";
    var data = JSON.stringify({'sentence': $scope.textModel});
    var headers = {'Content-Type': 'application/json; charset=utf-8'};
    $http.post('/api/is_correct', data=data, headers=headers).then(function(response){
      $scope.state = "idle";
      corrected_sentence = response['data']['sentence'];
      console.log(response)
      $scope.corrected_sentence = corrected_sentence;
    }).catch(function(error){
      $scope.state = "idle";
      console.log(error);
      alert('Error')
    });
  }
  function is_correct(probability){
    $scope.state = "correct";
    $scope.probability = probability;
  }

  function not_correct(probability){
    $scope.state = "incorrect";
    $scope.probability = 1- probability;
  }


});
