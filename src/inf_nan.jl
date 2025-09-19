
import JSON: show_json
import JSON.Serializations: CommonSerialization, StandardSerialization
import JSON: StructuralContext

struct NaNSerialization <: CommonSerialization end

function show_json(io::StructuralContext, ::NaNSerialization, f::AbstractFloat)
    if f==Inf
        Base.print(io, "Infinity")
    elseif f == -Inf
        Base.print(io, "-Infinity")
    elseif f == NaN
        Base.print(io, "NaN")
    else
        Base.print(io, f)
    end
end

jsonify(object::Any; allow_infnan=true) = sprint(show_json, allow_infnan ? NaNSerialization() : StandardSerialization(), object)
