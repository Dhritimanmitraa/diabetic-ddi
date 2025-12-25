import { motion } from 'framer-motion'
import { 
  Sparkles, ArrowRight, CheckCircle2, AlertCircle,
  Pill, Replace, Shuffle
} from 'lucide-react'

function AlternativesDisplay({ alternatives }) {
  if (!alternatives) return null

  const { 
    original_drug1, 
    original_drug2, 
    alternatives_for_drug1, 
    alternatives_for_drug2, 
    safe_combinations 
  } = alternatives

  return (
    <section className="max-w-4xl mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass rounded-3xl p-8"
      >
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-medical-400 to-medical-600 flex items-center justify-center">
            <Sparkles className="w-7 h-7 text-white" />
          </div>
          <div>
            <h2 className="font-display text-2xl font-bold text-white">
              Safe Alternatives
            </h2>
            <p className="text-slate-400">
              AI-recommended substitutes with no known interactions
            </p>
          </div>
        </div>

        {/* Alternatives for Drug 1 */}
        {alternatives_for_drug1 && alternatives_for_drug1.length > 0 && (
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <Replace className="w-5 h-5 text-medical-400" />
              <h3 className="font-semibold text-white">
                Alternatives for <span className="text-medical-400">{original_drug1?.name}</span>
              </h3>
            </div>
            <div className="grid gap-3">
              {alternatives_for_drug1.map((alt, index) => (
                <AlternativeCard key={index} alternative={alt} delay={index * 0.1} />
              ))}
            </div>
          </div>
        )}

        {/* Alternatives for Drug 2 */}
        {alternatives_for_drug2 && alternatives_for_drug2.length > 0 && (
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-4">
              <Replace className="w-5 h-5 text-medical-400" />
              <h3 className="font-semibold text-white">
                Alternatives for <span className="text-medical-400">{original_drug2?.name}</span>
              </h3>
            </div>
            <div className="grid gap-3">
              {alternatives_for_drug2.map((alt, index) => (
                <AlternativeCard key={index} alternative={alt} delay={0.3 + index * 0.1} />
              ))}
            </div>
          </div>
        )}

        {/* Safe Combinations */}
        {safe_combinations && safe_combinations.length > 0 && (
          <div className="pt-6 border-t border-slate-700/50">
            <div className="flex items-center gap-3 mb-4">
              <Shuffle className="w-5 h-5 text-medical-400" />
              <h3 className="font-semibold text-white">
                Recommended Safe Combinations
              </h3>
            </div>
            <div className="grid gap-3">
              {safe_combinations.slice(0, 5).map((combo, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 + index * 0.1 }}
                  className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl hover:bg-slate-800/50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-medical-500/10 rounded-lg">
                        <Pill className="w-4 h-4 text-medical-400" />
                      </div>
                      <span className="text-white font-medium">{combo.drug1?.name}</span>
                    </div>
                    <span className="text-slate-500">+</span>
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-medical-500/10 rounded-lg">
                        <Pill className="w-4 h-4 text-medical-400" />
                      </div>
                      <span className="text-white font-medium">{combo.drug2?.name}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5 text-green-400" />
                    <span className="text-green-400 text-sm font-medium">Safe</span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* Disclaimer */}
        <div className="mt-8 p-4 bg-warning-500/10 border border-warning-500/20 rounded-xl">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-warning-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-warning-400 font-medium mb-1">Important Notice</p>
              <p className="text-slate-400 text-sm">
                These alternatives are suggestions based on similar therapeutic effects. 
                Always consult with your healthcare provider or pharmacist before making 
                any changes to your medication regimen.
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  )
}

function AlternativeCard({ alternative, delay }) {
  const { drug, similarity_score, reason, has_interaction_with_other } = alternative

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay }}
      className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl hover:bg-slate-800/50 transition-colors group"
    >
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 rounded-xl bg-medical-500/10 flex items-center justify-center group-hover:scale-110 transition-transform">
          <Pill className="w-6 h-6 text-medical-400" />
        </div>
        <div>
          <p className="text-white font-medium">{drug?.name}</p>
          {drug?.drug_class && (
            <p className="text-slate-500 text-sm">{drug.drug_class}</p>
          )}
          {reason && (
            <p className="text-slate-400 text-xs mt-1">{reason}</p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Similarity score */}
        <div className="text-right">
          <p className="text-slate-400 text-xs mb-1">Similarity</p>
          <div className="flex items-center gap-2">
            <div className="w-16 h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-medical-400 rounded-full"
                style={{ width: `${similarity_score * 100}%` }}
              />
            </div>
            <span className="text-medical-400 text-sm font-mono">
              {Math.round(similarity_score * 100)}%
            </span>
          </div>
        </div>

        {/* Interaction status */}
        <div className={`px-3 py-1 rounded-full text-xs font-medium ${
          has_interaction_with_other 
            ? 'bg-yellow-500/10 text-yellow-400' 
            : 'bg-green-500/10 text-green-400'
        }`}>
          {has_interaction_with_other ? 'Minor interaction' : 'No interaction'}
        </div>

        <ArrowRight className="w-5 h-5 text-slate-500 group-hover:text-medical-400 group-hover:translate-x-1 transition-all" />
      </div>
    </motion.div>
  )
}

export default AlternativesDisplay

