module UMBridge

import HTTP
import JSON
import Base.Threads
using Parameters
using Sockets

# Make HTTP request following UM-Bridge protocol

struct HTTPModel
   name::String
   url::String
end

name(model::HTTPModel) = model.name
url(model::HTTPModel) = model.url

function check_response(response, expected_code)
    if response.status != expected_code
        error("Request failed with status code " * string(response.status) * " instead of " * string(expected_code))
    end
end

function check_parsed_response(parsed)
    if haskey(parsed, "error")
        error(parsed["error"]["type"] * ": " * parsed["error"]["message"])
    end
end

function evaluate(model, input, config = Dict())
    body = Dict(
        "name"   => name(model),
        "input"  => input,
        "config" => config
    )

    response = HTTP.request("POST", url(model) * "/Evaluate", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)

    return parsed["output"]
end

function gradient(model::HTTPModel, out_wrt, in_wrt, input, sens, config = Dict())
    body = Dict(
        "name" =>name(model),
        "outWrt" => out_wrt,
        "inWrt" => in_wrt,
        "input" => input,
        "sens" => sens,
        "config" => config
    )

    response = HTTP.request("POST", url(model) * "/Gradient", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["output"]
end

function apply_jacobian(model::HTTPModel, out_wrt, in_wrt, input, vec, config = Dict())
    body = Dict(
        "name" =>name(model),
        "outWrt" => out_wrt,
        "inWrt" => in_wrt,
        "input" => input,
        "vec" => vec,
        "config" => config
    )

    response = HTTP.request("POST", url(model) * "/ApplyJacobian", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["output"]
end

function apply_hessian(model::HTTPModel, out_wrt, in_wrt1, in_wrt2, input, vec, sens, config = Dict())
    body = Dict(
        "name"   => name(model),
        "outWrt" => out_wrt,
        "inWrt1" => in_wrt1,
        "inWrt2" => in_wrt2,
        "input" => input,
        "vec" => vec,
        "sens" => sens,
        "config" => config
    )

    response = HTTP.request("POST", url(model) * "/ApplyHessian", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["output"]
end

function protocol_version_supported(model::HTTPModel)
    response = HTTP.request("GET", url(model) * "/Info")
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["protocolVersion"] == 1.0
end

function get_models(model::HTTPModel)
    response = HTTP.request("GET", url(model) * "/Info")
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["models"]
end

function model_input_sizes(model::HTTPModel, config = Dict())
    body = Dict(
        "name"   => name(model),
        "config" => config
    )
    response = HTTP.request("POST", url(model) * "/InputSizes", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["inputSizes"]
end

function model_output_sizes(model::HTTPModel, config = Dict())
    body = Dict(
        "name"   => name(model),
        "config" => config
    )
    response = HTTP.request("POST", url(model) * "/OutputSizes", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["outputSizes"]
end

function supports_evaluate(model::HTTPModel)
    body = Dict(
        "name" => name(model)
    )
    response = HTTP.request("POST", url(model) * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["Evaluate"]
end

function supports_gradient(model::HTTPModel)
    body = Dict(
        "name" => name(model)
    )
    response = HTTP.request("POST", url(model) * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["Gradient"]
end

function supports_apply_jacobian(model::HTTPModel)
    body = Dict(
        "name" => name(model)
    )
    response = HTTP.request("POST", url(model) * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["ApplyJacobian"]
end

function supports_apply_hessian(model::HTTPModel)
    body = Dict(
        "name" => name(model)
    )
    response = HTTP.request("POST", url(model) * "/ModelInfo", body=JSON.json(body))
    check_response(response, 200)
    parsed = JSON.parse(String(response.body))
    check_parsed_response(parsed)
    return parsed["support"]["ApplyHessian"]
end

@with_kw mutable struct Model
    name::String
    inputSizes::AbstractArray
    outputSizes::AbstractArray
    supportsEvaluate::Bool = true
    supportsGradient::Bool = false
    supportsJacobian::Bool = false
    supportsHessian::Bool  = false
    evaluate::Function = (input::Any, config::Any) -> (error("Evaluate: Not implemented"))
    gradient::Function = (outWrt::Any, inWrt::Any, input::Any, sens::Any, config::Any) -> (error("Gradient: Not implemented"))
    applyJacobian::Function = (outWrt::Any, inWrt::Any, input::Any, vec::Any, config::Any) -> (error("Apply Jacobian: Not implemented"))
    applyHessian::Function = (outWrt::Any, inWrt1::Any, inWrt2::Any, input::Any, sens::Any, vec::Any, config::Any) -> (error("Apply Hessian: Not implemented"))
end

name(model::Model) = model.name
inputSizes(model::Model) = model.inputSizes
outputSizes(model::Model) = model.outputSizes

supportsEvaluate(model::Model) = model.supportsEvaluate
supportsGradient(model::Model) = model.supportsGradient
supportsJacobian(model::Model) = model.supportsJacobian
supportsHessian(model::Model) = model.supportsHessian

function define_evaluate(model::Model, model_evaluate)
    model.evaluate = model_evaluate
end

function define_gradient(model::Model, model_gradient)
    model.gradient = model_gradient
end

function define_applyjacobian(model::Model, model_jacobian)
    model.applyJacobian = model_jacobian
end

function define_applyhessian(model::Model, model_hessian)
    model.applyHessian = model_hessian
end

function get_model_from_name(models::Vector, model_name::String)
    for model in models
        if name(model) == model_name
            return model
        end
    end
    return nothing
end

function inputRequest(models::Vector)
    function handler(request::HTTP.Request)
        model_name = JSON.parse(String(request.body))["name"]
        model = get_model_from_name(models, model_name)
	if model == nothing
		body = Dict(
			"error" => Dict(
				"type" => "ModelNotFound",
				"message" => "Model name not found"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end

        body = Dict(
            "inputSizes" => inputSizes(model)
        )
        return HTTP.Response(JSON.json(body))
    end
    return handler
end

function outputRequest(models::Vector)
    function handler(request::HTTP.Request)
        model_name = JSON.parse(String(request.body))["name"]
        model = get_model_from_name(models, model_name)
	if model == nothing
		body = Dict(
			"error" => Dict(
				"type" => "ModelNotFound",
				"message" => "Model name not found"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end

        body = Dict(
            "outputSizes" => outputSizes(model)
        )
        return HTTP.Response(JSON.json(body))
    end
    return handler
end

function infoRequest(models::Vector)
     function handler(request::HTTP.Request)
		body = Dict(
			"protocolVersion" => 1.0,
			"models" => [model.name for model in models]
		)
        return HTTP.Response(JSON.json(body))
    end
    return handler
end

function modelinfoRequest(models::Vector)
     function handler(request::HTTP.Request)
        model_name = JSON.parse(String(request.body))["name"]
        model = get_model_from_name(models, model_name)
	if model == nothing
		body = Dict(
			"error" => Dict(
				"type" => "ModelNotFound",
				"message" => "Model name not found"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end

        body = Dict( "support" => Dict(
            "Evaluate" => supportsEvaluate(model),
            "Gradient" => supportsGradient(model),
            "ApplyJacobian" => supportsJacobian(model),
            "ApplyHessian" => supportsHessian(model)
        ))
        return HTTP.Response(JSON.json(body))
    end
    return handler
end

function evaluateRequest(models::Vector)
    function handler(request::HTTP.Request)
	# Parse the JSON body
	parsed_body = JSON.parse(String(request.body))
	# Extract the model name directly from parsed_body
	model_name = parsed_body["name"]
	model = get_model_from_name(models, model_name)
	if model == nothing
		body = Dict(
			"error" => Dict(
				"type" => "ModelNotFound",
				"message" => "Model name not found"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end

	# Extract inputs and check
        model_parameters = parsed_body["input"]
	if length(model_parameters) != length(inputSizes(model))
		body = Dict(
			"error" => Dict(
				"type" => "InvalidInput",
				"message" => "Invalid input"
			)
		)
		return HTTP.Response(400, JSON.json(body))

	end
	for i in 1:length(model_parameters)
		if length(model_parameters[i]) != inputSizes(model)[i]
			body = Dict(
				    "error" => Dict(
						    "type" => "InvalidInput",
						    "message" => "Invalid input"
						    )
				    )
			return HTTP.Response(400, JSON.json(body))
		end
	end
        if !supportsEvaluate(model)
		body = Dict(
			"error" => Dict(
				"type" => "UnsupportedFeature",
				"message" => "Unsupported feature"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end


	# Extract config
	if haskey(parsed_body,"config")
		model_config = parsed_body["config"]
	else
		model_config = Dict()
	end
	# Apply model's evaluate
	output = model.evaluate(model_parameters, model_config)
	if length(output) != length(outputSizes(model))
		body = Dict(
			"error" => Dict(
				"type" => "InvalidOutput",
				"message" => "Invalid output"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end
	for i in 1:length(output)
		if length(output[i]) != outputSizes(model)[i]
			body = Dict(
				    "error" => Dict(
						    "type" => "InvalidOutput",
						    "message" => "Invalid output"
						    )
				    )
			return HTTP.Response(400, JSON.json(body))
		end
	end
        body = Dict(
		    "output" => output
		)
        return HTTP.Response(JSON.json(body))
    end
    return handler
end

function gradientRequest(models::Vector)
     function handler(request::HTTP.Request)
	parsed_body = JSON.parse(String(request.body))
        model_name = parsed_body["name"]
        model = get_model_from_name(models, model_name)
        if model == nothing
		body = Dict(
			"error" => Dict(
				"type" => "ModelNotFound",
				"message" => "Model name not found"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end
	if !supportsGradient(model)
		body = Dict(
			"error" => Dict(
				"type" => "UnsupportedFeature",
				"message" => "Unsupported feature"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end

	model_inWrt = parsed_body["inWrt"] + 1 # account for julia indices starting at 1
	if model_inWrt < 1 || model_inWrt > length(inputSizes(model))
		body = Dict(
			"error" => Dict(
				"type" => "InvalidInput",
				"message" => "Invalid inWrt index! Expected between 0 and  and number of inputs minus one, but got " * string(model_inWrt - 1)
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end
	model_outWrt = parsed_body["outWrt"] + 1 # account for julia indices starting at 1
	if model_outWrt < 1 || model_outWrt > length(inputSizes(model))
		body = Dict(
			"error" => Dict(
				"type" => "InvalidInput",
				"message" => "Invalid outWrt index! Expected between 0 and  and number of inputs minus one, but got " * string(model_outWrt - 1)
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end
	model_sens = parsed_body["sens"]
        model_parameters = parsed_body["input"]
	if length(model_parameters) != length(inputSizes(model))
		body = Dict(
			"error" => Dict(
				"type" => "InvalidInput",
				"message" => "Invalid input"
			)
		)
		return HTTP.Response(400, JSON.json(body))

	end
	for i in 1:length(model_parameters)
		if length(model_parameters[i]) != inputSizes(model)[i]
			body = Dict(
				    "error" => Dict(
						    "type" => "InvalidInput",
						    "message" => "Invalid input"
						    )
				    )
			return HTTP.Response(400, JSON.json(body))
		end
	end
        
	if haskey(parsed_body, "config")
		model_config = parsed_body["config"]
	else
		model_config = Dict()
	end
	# Apply model's gradient
	output = model.gradient(model_outWrt, model_inWrt, model_parameters, model_sens, model_config)

        body = Dict(
		    "output" => output
        )
        return HTTP.Response(JSON.json(body))
    end
    return handler
end

function applyJacobianRequest(models::Vector)
     function handler(request::HTTP.Request)
	parsed_body = JSON.parse(String(request.body))
        model_name = parsed_body["name"]
        model = get_model_from_name(models, model_name)
        if model == nothing
		body = Dict(
			"error" => Dict(
				"type" => "ModelNotFound",
				"message" => "Model name not found"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end
	if !supportsJacobian(model)
		body = Dict(
			"error" => Dict(
				"type" => "UnsupportedFeature",
				"message" => "Unsupported feature"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end


	model_inWrt = parsed_body["inWrt"]
        model_outWrt = parsed_body["outWrt"]
        model_vec = parsed_body["vec"]
        model_parameters = parsed_body["input"]
        
	if haskey(parsed_body, "config")
		model_config = parsed_body["config"]
        else
		model_config = Dict()
	end
	# Apply model's Jacobian
	output = model.applyJacobian(model_outWrt, model_inWrt, model_parameters, model_vec, model_config)
	body = Dict(
		    "output" => output
        )
        return HTTP.Response(JSON.json(body))
    end
    return handler
end

function applyHessianRequest(models::Vector)
     function handler(request::HTTP.Request)
	parsed_body = JSON.parse(String(request.body))
        model_name = parsed_body["name"]
        model = get_model_from_name(models, model_name)
        if model == nothing
		body = Dict(
			"error" => Dict(
				"type" => "ModelNotFound",
				"message" => "Model name not found"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end
	if !supportsHessian(model)
		body = Dict(
			"error" => Dict(
				"type" => "UnsupportedFeature",
				"message" => "Unsupported feature"
			)
		)
		return HTTP.Response(400, JSON.json(body))
	end

	model_inWrt1 = parsed_body["inWrt1"]
        model_inWrt2 = parsed_body["inWrt2"]
        model_outWrt = parsed_body["outWrt"]
        model_sens = parsed_body["sens"]
        model_vec = parsed_body["vec"]
        model_parameters = parsed_body["input"]
        
	if haskey(parsed_body, "config")
		model_config = parsed_body["config"]
	else
		model_config = Dict()
	end
	# Apply model's Hessian
	output = model.applyHessian(model_outWrt, model_inWrt1, model_inWrt2, model_parameters, model_sens, model_vec, model_config)
	
        body = Dict(
		    "output" => output
        )
        return HTTP.Response(JSON.json(body))
    end
    return handler
end

function serve_models(models::Vector, port=4242, max_workers=1)
    router = HTTP.Router()
    HTTP.register!(router, "POST", "/InputSizes", inputRequest(models))
    HTTP.register!(router, "POST", "/OutputSizes", outputRequest(models))
    HTTP.register!(router, "GET", "/Info", infoRequest(models))
    HTTP.register!(router, "POST", "/ModelInfo", modelinfoRequest(models))
    HTTP.register!(router, "POST", "/Evaluate", evaluateRequest(models))
    HTTP.register!(router, "POST", "/Gradient", gradientRequest(models))
    HTTP.register!(router, "POST", "/ApplyJacobian", applyJacobianRequest(models))
    HTTP.register!(router, "POST", "/ApplyHessian", applyHessianRequest(models))
    server = HTTP.serve(router, ip"0.0.0.0",  port)
end

end
