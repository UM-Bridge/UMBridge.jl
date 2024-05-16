using UMBridge
using HTTP
using JSON
using Test

const port = 5252

function testmodel_supported_operations(model)
    all([
        UMBridge.supports_evaluate(model),
        !UMBridge.supports_gradient(model),
        !UMBridge.supports_apply_jacobian(model),
        !UMBridge.supports_apply_hessian(model)
   ])
end

function testserver(models)
    # Serve model
    # port = port
    server = UMBridge.serve_models(models, port)
    # Shut down server
    close(server)
    return true
end

function testclientserverpair(models, httpmodel)
    server = UMBridge.serve_models(models, port)
    input = UMBridge.model_input_sizes(httpmodel)[1] == 1
    output = UMBridge.model_output_sizes(httpmodel)[1] == 1
    close(server)
    return all([input, output])
end

function testserver_sizes(models)
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "config" => Dict()
    )
    response_input  = UMBridge.inputRequest(models)(HTTP.Request("POST", "/InputSizes", [], JSON.json(body)))
    response_output = UMBridge.outputRequest(models)(HTTP.Request("POST", "/OutputSizes", [], JSON.json(body)))
    all([response_input.status == 200, response_output.status == 200])
end

function testserver_info(models)
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "config" => Dict()
    )
    response_input  = UMBridge.infoRequest(models)(HTTP.Request("GET", "/Info", [], JSON.json(body)))
    return response_input.status == 200
end

function testserver_evaluate(models)
    UMBridge.define_evaluate(models[1], (input, config) -> (input))
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "input" => [1.0],
        "config" => Dict()
    )
    response_input  = UMBridge.evaluateRequest(models)(HTTP.Request("POST", "/Evaluate", [], JSON.json(body)))
    return response_input.status == 200
end

function testserver_gradient(models)
    models[1].supportsGradient = true
    UMBridge.define_gradient(models[1], (outWrt, inWrt, input, sens, config) -> ([1]))
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "inWrt" => [1],
        "outWrt" => [1],
        "sens" => [1],
        "input" => [1],
        "config" => Dict()
    )
    response_input  = UMBridge.gradientRequest(models)(HTTP.Request("POST", "/Gradient", [], JSON.json(body)))
    return response_input.status == 200
end

function testserver_jacobian(models)
    models[1].supportsJacobian = true
    UMBridge.define_applyjacobian(models[1], (outWrt, inWrt, input, vec, config) -> ([1]))
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "inWrt" => [1],
        "outWrt" => [1],
        "input" => [1],
        "vec" => [1],
        "config" => Dict()
    )
    response_input  = UMBridge.applyJacobianRequest(models)(HTTP.Request("POST", "/ApplyJacobian", [], JSON.json(body)))
    return response_input.status == 200
end

function testserver_hessian(models)
    models[1].supportsHessian = true
    UMBridge.define_applyhessian(models[1], (outWrt, inWrt1, inWrt2, input, sens, vec, config) -> ([1]))
    body = Dict(
        "name" => UMBridge.name(models[1]),
        "inWrt1" => [1],
        "inWrt2" => [1],
        "outWrt" => [1],
        "input" => [1],
        "vec" => [1],
        "sens" => [1],
        "config" => Dict()
    )
    response_input  = UMBridge.applyHessianRequest(models)(HTTP.Request("POST", "/ApplyHessian", [], JSON.json(body)))
    return response_input.status == 200
end

@testset "UMBridge.jl" begin

    # Test client
    httpmodel = UMBridge.HTTPModel("forward", "https://testmodel.linusseelinger.de")
    @test UMBridge.evaluate(httpmodel, [[4]], Dict())[1][1] == 8.0
    @test UMBridge.protocol_version_supported(httpmodel)
    @test UMBridge.get_models(httpmodel)[1] == "forward"
    @test UMBridge.model_input_sizes(httpmodel)[1] == 1
    @test UMBridge.model_output_sizes(httpmodel)[1] == 1
    @test testmodel_supported_operations(httpmodel)

    # Test server
    modelinput = UMBridge.model_input_sizes(httpmodel)
    modeloutput = UMBridge.model_output_sizes(httpmodel)
    models = [UMBridge.Model(name = "forward", inputSizes = modelinput, outputSizes = modeloutput)]
    @test testserver(models)
    @test testclientserverpair(models,httpmodel)
    @test testserver_sizes(models)
    @test testserver_info(models)
    @test testserver_evaluate(models)
    @test testserver_gradient(models)
    @test testserver_jacobian(models)
    @test testserver_hessian(models)

end

@testset "Test Connection" begin
    
    model=UMBridge.HTTPModel("posterior", "https://benchmark-analytic-funnel.linusseelinger.de")
    httpValue = UMBridge.evaluate(model, [[1, 3]], Dict())[1][1]
    exactValue = -5.147502395904501;
    @test isapprox(httpValue, exactValue, rtol=1e-14)
    
end

@testset "Test Apply Jacobian" begin
    
    model=UMBridge.HTTPModel("posterior", "https://benchmark-analytic-funnel.linusseelinger.de")
    httpValue = UMBridge.apply_jacobian(model, 0, 0, [[1, 3]], [1, 4])[1]
    exactValue = -3.370206919896928;
    @test isapprox(httpValue, exactValue, rtol=1e-14)
    
end

@testset "Test Gradient" begin
    
    model=UMBridge.HTTPModel("posterior", "https://benchmark-analytic-funnel.linusseelinger.de")
    httpValue = UMBridge.gradient(model, 0, 0, [[1, 3]], [12])[1][1]
    exactValue = 12.53215648992455;
    @test isapprox(httpValue, exactValue, rtol=1e-14)
    
end


using Zygote  # For automatic differentiation

# Define model for 1D function f(x) = x^2 that uses autodiff
model = UMBridge.Model(
    name = "quadratic",
    inputSizes = [1],
    outputSizes = [1],
    supportsGradient = true,
    evaluate = (input, config) -> input[1]^2,
    gradient = (outWrt, inWrt, input, sens, config) -> (Zygote.gradient(x -> x^2, input[1])[1] * sens[1])
)

function testserver_autodiff_gradient(models)
    input = [2.0]  # Example input
    sens = [1.0]

    body = Dict(
        "name" => UMBridge.name(models[1]),
        "inWrt" => [1],
        "outWrt" => [1],
        "sens" => sens,
        "input" => input,
        "config" => Dict()
    )

    # Make gradient request
    response_input = UMBridge.gradientRequest(models)(HTTP.Request("POST", "/Gradient", [], JSON.json(body)))
    expected_gradient = models[1].gradient(1, 1, input, sens, Dict())
    
    # Verify the gradient application result 
    return response_input.status == 200 && JSON.parse(String(response_input.body))["output"] == expected_gradient
end

models = [model]
@testset "UMBridge Autodiff Test" begin
    @test testserver_autodiff_gradient(models)
end


# Define model for 2D function f(x) = x^2 that uses autodiff
model = UMBridge.Model(
    name = "quadratic2D",
    inputSizes = [2],
    outputSizes = [2],
    supportsJacobian = true,
    evaluate = (input, config) -> [input[1]^2, input[2]^2],
    applyJacobian = (outWrt, inWrt, input, vec, config) -> Zygote.jacobian(x -> [x[1]^2, x[2]^2], input)[1] * vec
)

function testserver_autodiff_jacobian(models)
    input = [2.0, 3.0]  # example 2D input
    vec = [1.0, 1.0]    # vector for Jacobian application

    body = Dict(
        "name" => UMBridge.name(models[1]),
        "outWrt" => [1, 2],
        "inWrt" => [1, 2],
        "input" => input,
        "vec" => vec,
        "config" => Dict()
    )

     # Make jacobian request
    response_input = UMBridge.applyJacobianRequest(models)(HTTP.Request("POST", "/ApplyJacobian", [], JSON.json(body)))
    expected_jacobian_application = models[1].applyJacobian(1, 1, input, vec, Dict())

    # Print results
    println("Expected Jacobian application result: ", expected_jacobian_application)
    println("Response from server: ", JSON.parse(String(response_input.body))["output"])

    # Verify the jacobian application result
    is_correct = response_input.status == 200 && all(JSON.parse(String(response_input.body))["output"][i] == expected_jacobian_application[i] for i in 1:length(input))
    println("Test passed: ", is_correct)
    return is_correct
end


@testset "UMBridge 2D Apply Jacobian Test" begin
    models_2D = [model]
    @test testserver_autodiff_jacobian(models_2D)
end
